"""Part B: RFM segmentation and a leakage-aware lapse model."""
from pathlib import Path
import numpy as np, pandas as pd, joblib
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.base import clone
from sklearn.utils import resample
from scipy.sparse import issparse
ROOT=Path(__file__).resolve().parents[1]; CLEAN=ROOT/"data/clean"; OUT=ROOT/"outputs"
SNAPSHOT=pd.Timestamp("2025-10-01")
def load():
    c=pd.read_csv(CLEAN/"customers.csv"); o=pd.read_csv(CLEAN/"orders.csv")
    o["order_ts"]=pd.to_datetime(o.order_ts)
    return c,o
def build_rfm_table():

    # 1. Locate cleaned data
    ROOT = Path(__file__).resolve().parent.parent
    CLEAN = ROOT / "data" / "clean"

    orders = pd.read_csv(CLEAN / "orders.csv")

    # 2. Keep only completed orders
    completed_orders = orders[
        orders["status"] == "completed"
    ].copy()

    # 3. Convert columns to appropriate data types
    completed_orders["order_ts"] = pd.to_datetime(
        completed_orders["order_ts"],
        errors="coerce"
    )

    completed_orders["order_amount"] = pd.to_numeric(
        completed_orders["order_amount"],
        errors="coerce"
    )

    # 4. Define snapshot moment
    snapshot_date = pd.Timestamp("2025-10-01 00:00:00")

    # Exclude invalid timestamps and orders after the snapshot
    completed_orders = completed_orders[
        completed_orders["order_ts"].notna()
        & (completed_orders["order_ts"] <= snapshot_date)
    ].copy()

    # 5. Group orders by customer and calculate RFM
    rfm = (
        completed_orders
        .groupby("customer_id")
        .agg(
            last_order_ts=("order_ts", "max"),
            frequency=("order_id", "nunique"),
            monetary=("order_amount", "sum")
        )
        .reset_index()
    )

    # 6. Calculate recency in whole days
    rfm["recency"] = (
        snapshot_date - rfm["last_order_ts"]
    ).dt.days

    # 7. Select required columns
    rfm = rfm[
        ["customer_id", "recency", "frequency", "monetary"]
    ]

    rfm["monetary"] = rfm["monetary"].round(2)

    # 8. Save RFM table
    rfm.to_csv(CLEAN / "rfm.csv", index=False)

    return rfm
def actual_rfm(rfm):

    rfm_summary = pd.DataFrame({
        "Metric": [
            "Customer count",
            "Recency median",
            "Recency minimum",
            "Recency maximum",
            "Frequency mean",
            "Frequency maximum",
            "Monetary mean",
            "Monetary maximum"
        ],
        "Actual value": [
            len(rfm),
            rfm["recency"].median(),
            rfm["recency"].min(),
            rfm["recency"].max(),
            rfm["frequency"].mean(),
            rfm["frequency"].max(),
            rfm["monetary"].mean(),
            rfm["monetary"].max()
        ]
    })

    print("\n=== RFM SUMMARY ===")
    print(rfm_summary.round(2).to_string(index=False))

    return rfm_summary
def rfm_and_segments(k=4):
    c,o=load(); done=o[o.status.eq("completed")].copy()
    rfm=done.groupby("customer_id").agg(last_order=("order_ts","max"),frequency=("order_id","nunique"),monetary=("order_amount","sum")).reset_index()
    rfm["recency"]=(SNAPSHOT-rfm.last_order).dt.days; rfm=rfm.drop(columns="last_order")
    scaler=StandardScaler(); X=scaler.fit_transform(rfm[["recency","frequency","monetary"]])
    diagnostics=[]
    from sklearn.metrics import silhouette_score
    for n in range(2,9):
        km=KMeans(n_clusters=n,n_init=10,random_state=42).fit(X)
        diagnostics.append({"k":n,"inertia":km.inertia_,"silhouette":silhouette_score(X,km.labels_)})
    km=KMeans(n_clusters=k,n_init=10,random_state=42); rfm["segment_id"]=km.fit_predict(X)
    profile=rfm.groupby("segment_id").agg(size=("customer_id","size"),recency=("recency","mean"),frequency=("frequency","mean"),monetary=("monetary","mean")).reset_index()
    order=profile.sort_values("monetary",ascending=False).segment_id.tolist()
    names=["Premium" if i==order[0] else ("Frequent" if profile.set_index("segment_id").loc[i,"frequency"]>=profile.frequency.median() else "At Risk") for i in profile.segment_id]
    name_map=dict(zip(profile.segment_id,names))
    rfm["segment_name"]=rfm.segment_id.map(name_map); rfm["is_premium"]=rfm.segment_id.eq(order[0])
    OUT.mkdir(exist_ok=True)
    rfm[["customer_id","recency","frequency","monetary","segment_id","segment_name","is_premium"]].to_csv(OUT/"customer_segments.csv",index=False)
    pca=PCA(n_components=2).fit(X)
    return rfm,pd.DataFrame(diagnostics),profile,pca.explained_variance_ratio_

def build_lapse_model():

    c, o = load()

    cutoff = pd.Timestamp("2025-09-01")
    end = pd.Timestamp("2025-10-01")

    # =====================================================
    # 1. FEATURE ENGINEERING USING PRE-CUTOFF DATA ONLY
    # =====================================================

    hist = o[o.order_ts < cutoff].copy()
    done = hist[hist.status.eq("completed")]

    cohort = done.groupby("customer_id").agg(
        last_order=("order_ts", "max"),
        frequency=("order_id", "nunique"),
        monetary=("order_amount", "sum"),
        avg_discount=("discount_pct", "mean"),
        avg_delivery=("delivery_minutes", "mean")
    ).reset_index()

    cohort["recency"] = (
        cutoff - cohort["last_order"]
    ).dt.days

    cohort = cohort.drop(columns="last_order")

    all_final = (
        hist[hist.status.isin(["completed", "cancelled"])]
        .groupby("customer_id")
        .status
        .agg(
            total="count",
            cancelled=lambda x: (x == "cancelled").sum()
        )
        .reset_index()
    )

    cohort = cohort.merge(
        all_final, on="customer_id", how="left"
    )

    cohort["cancellation_rate"] = (
        cohort["cancelled"] / cohort["total"]
    )

    # Favourite product category
    products = pd.read_csv(CLEAN / "products.csv")

    cats = (
        done.merge(
            products[["product_id", "category"]],
            on="product_id"
        )
        .groupby(["customer_id", "category"])
        .size()
        .rename("n")
        .reset_index()
    )

    fav = (
        cats.sort_values(
            ["customer_id", "n", "category"],
            ascending=[True, False, True]
        )
        .drop_duplicates("customer_id")
        .rename(columns={"category": "favourite_category"})
        [["customer_id", "favourite_category"]]
    )

    cohort = (
        cohort.merge(fav, on="customer_id", how="left")
        .merge(c, on="customer_id", how="left")
    )

    cohort["signup_date"] = pd.to_datetime(
        cohort["signup_date"]
    )

    cohort["tenure_days"] = (
        cutoff - cohort["signup_date"]
    ).dt.days

    # September lapse label
    post = (
        o[
            (o.order_ts >= cutoff)
            & (o.order_ts < end)
            & o.status.eq("completed")
        ]
        .groupby("customer_id")
        .size()
    )

    cohort["lapsed"] = (
        ~cohort.customer_id.isin(post.index)
    ).astype(int)

    cohort = cohort.drop(
        columns=["signup_date", "total", "cancelled"]
    )

    # =====================================================
    # 2. FEATURES AND TARGET
    # =====================================================

    features = [
        "recency", "frequency", "monetary",
        "avg_discount", "cancellation_rate",
        "avg_delivery", "favourite_category",
        "city", "age", "gender", "tenure_days"
    ]

    numeric = [
        "recency", "frequency", "monetary",
        "avg_discount", "cancellation_rate",
        "avg_delivery", "age", "tenure_days"
    ]

    categorical = [
        "favourite_category", "city", "gender"
    ]

    X = cohort[features].copy()
    y = cohort["lapsed"].copy()

    # Keep customer IDs aligned with the original row indices
    customer_ids = cohort["customer_id"].copy()

    print("\n=== MISSING VALUE CHECK: RAW FEATURES ===")
    print(X.isnull().sum())

    # =====================================================
    # 3. TRAIN / TEST SPLIT
    # =====================================================

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )

    # =====================================================
    # 4. PREPROCESSING PIPELINE
    # Fit imputer, scaler and encoder on training rows only
    # =====================================================

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(
            strategy="constant",
            fill_value="unknown"
        )),
        ("encoder", OneHotEncoder(
            drop="first",
            handle_unknown="ignore"
        ))
    ])

    prep = ColumnTransformer([
        ("num", numeric_pipeline, numeric),
        ("cat", categorical_pipeline, categorical)
    ])

    # =====================================================
    # 5. GRID SEARCH: LOGISTIC REGRESSION AND RANDOM FOREST
    # =====================================================

    models = {}

    model_configs = [
        (
            "logistic",
            LogisticRegression(max_iter=1000),
            {"model__C": [0.01, 0.1, 1, 10, 100]}
        ),
        (
            "forest",
            RandomForestClassifier(
                n_estimators=100,
                random_state=42
            ),
            {
                "model__max_depth": [3, 5, 8],
                "model__min_samples_leaf": [5, 10, 20]
            }
        )
    ]

    for name, estimator, grid in model_configs:

        pipe = Pipeline([
            ("prep", prep),
            ("model", estimator)
        ])

        search = GridSearchCV(
            pipe,
            grid,
            cv=5,
            scoring="f1",
            n_jobs=-1
        )

        search.fit(X_train, y_train)

        pred = search.predict(X_test)

        models[name] = {
            "best_params": search.best_params_,
            "cv_f1": search.best_score_,
            "precision": precision_score(
                y_test, pred, zero_division=0
            ),
            "recall": recall_score(
                y_test, pred, zero_division=0
            ),
            "f1": f1_score(
                y_test, pred, zero_division=0
            ),
            "confusion_matrix": confusion_matrix(
                y_test, pred, labels=[0, 1]
            ).tolist(),
            "search": search
        }

    # =====================================================
    # 6. BASELINE: PREDICT EVERY CUSTOMER AS NOT LAPSED
    # =====================================================

    baseline_pred = np.zeros(len(y_test), dtype=int)

    baseline = {
        "accuracy": accuracy_score(y_test, baseline_pred),
        "precision": precision_score(
            y_test, baseline_pred, zero_division=0
        ),
        "recall": recall_score(
            y_test, baseline_pred, zero_division=0
        ),
        "f1": f1_score(
            y_test, baseline_pred, zero_division=0
        ),
        "confusion_matrix": confusion_matrix(
            y_test, baseline_pred, labels=[0, 1]
        ).tolist()
    }

    print("\n=== BASELINE RESULTS ===")
    print(baseline)

    # =====================================================
    # 7. REPORT TUNED MODEL RESULTS
    # =====================================================

    for name, result in models.items():

        print(f"\n=== {name.upper()} RESULTS ===")
        print("Best parameters:", result["best_params"])
        print("Best CV F1:", round(result["cv_f1"], 3))
        print("Confusion matrix:")
        print(result["confusion_matrix"])
        print("Precision:", round(result["precision"], 3))
        print("Recall:", round(result["recall"], 3))
        print("F1:", round(result["f1"], 3))

    # =====================================================
    # 8. FREQUENCY MEAN COMPARISON
    # =====================================================

    best_logistic = models["logistic"]["search"].best_estimator_

    fitted_numeric_pipeline = (
        best_logistic.named_steps["prep"]
        .named_transformers_["num"]
    )

    frequency_index = numeric.index("frequency")

    train_frequency_mean = (
        fitted_numeric_pipeline
        .named_steps["scaler"]
        .mean_[frequency_index]
    )

    full_cohort_frequency_mean = X["frequency"].mean()

    print("\n=== FREQUENCY MEAN COMPARISON ===")
    print("Training scaler frequency mean:", train_frequency_mean)
    print("Full-cohort frequency mean:", full_cohort_frequency_mean)

    # =====================================================
    # 9. CONFIRM NO MISSING VALUES AFTER PREPROCESSING
    # =====================================================

    fitted_prep = best_logistic.named_steps["prep"]

    transformed_train = fitted_prep.transform(X_train)
    transformed_test = fitted_prep.transform(X_test)

    if issparse(transformed_train):
        train_has_missing = np.isnan(transformed_train.data).any()
    else:
        train_has_missing = np.isnan(transformed_train).any()

    if issparse(transformed_test):
        test_has_missing = np.isnan(transformed_test.data).any()
    else:
        test_has_missing = np.isnan(transformed_test).any()

    assert not train_has_missing
    assert not test_has_missing

    print("\nNo missing values in transformed training features:", not train_has_missing)
    print("No missing values in transformed test features:", not test_has_missing)

    # =====================================================
    # 10. RANDOM FOREST FEATURE IMPORTANCE
    # =====================================================

    best_forest = models["forest"]["search"].best_estimator_

    feature_names = (
        best_forest.named_steps["prep"]
        .get_feature_names_out()
    )

    importances = (
        best_forest.named_steps["model"]
        .feature_importances_
    )

    feature_importance = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    }).sort_values("Importance", ascending=False)

    print("\n=== TOP 5 RANDOM FOREST FEATURES ===")
    print(feature_importance.head(5).to_string(index=False))

    # =====================================================
    # 11. LOGISTIC REGRESSION COEFFICIENTS
    # =====================================================

    logistic_features = (
        best_logistic.named_steps["prep"]
        .get_feature_names_out()
    )

    coefficients = (
        best_logistic.named_steps["model"].coef_[0]
    )

    coefficient_table = pd.DataFrame({
        "Feature": logistic_features,
        "Coefficient": coefficients
    })

    coefficient_table["Direction"] = np.where(
        coefficient_table["Coefficient"] > 0,
        "Positive",
        "Negative"
    )

    coefficient_table["Abs_Coefficient"] = (
        coefficient_table["Coefficient"].abs()
    )

    coefficient_table = coefficient_table.sort_values(
        "Abs_Coefficient", ascending=False
    )

    print("\n=== LOGISTIC REGRESSION COEFFICIENTS ===")
    print(coefficient_table.head(10).to_string(index=False))

    # =====================================================
    # 12. CLASS BALANCE EXPERIMENT
    # Oversample training rows only
    # =====================================================

    balanced_train = X_train.copy()
    balanced_train["lapsed"] = y_train

    majority = balanced_train[
        balanced_train["lapsed"] == 0
    ]

    minority = balanced_train[
        balanced_train["lapsed"] == 1
    ]

    additional_count = len(majority) - len(minority)

    if additional_count < 0:
        raise ValueError(
            "Lapsed class is not the minority class."
        )

    if additional_count > 0:
        extra_minority = resample(
            minority,
            replace=True,
            n_samples=additional_count,
            random_state=42
        )
    else:
        extra_minority = minority.iloc[0:0]

    balanced_data = pd.concat([
        balanced_train,
        extra_minority
    ])

    X_balanced = balanced_data.drop(columns="lapsed")
    y_balanced = balanced_data["lapsed"]

    print("\n=== BALANCED TRAINING COUNTS ===")
    print(y_balanced.value_counts().sort_index())

    # Refit the best logistic regression pipeline
    balanced_logistic = clone(best_logistic)

    balanced_logistic.fit(X_balanced, y_balanced)

    balanced_pred = balanced_logistic.predict(X_test)

    balanced_results = {
        "precision": precision_score(
            y_test, balanced_pred, zero_division=0
        ),
        "recall": recall_score(
            y_test, balanced_pred, zero_division=0
        ),
        "f1": f1_score(
            y_test, balanced_pred, zero_division=0
        )
    }

    print("\n=== ORIGINAL LOGISTIC REGRESSION ===")
    print({
        "precision": models["logistic"]["precision"],
        "recall": models["logistic"]["recall"],
        "f1": models["logistic"]["f1"]
    })

    print("\n=== BALANCED LOGISTIC REGRESSION ===")
    print(balanced_results)

    # =====================================================
    # 13. SELECT AND SAVE RANDOM FOREST MODEL
    # =====================================================

    selected = best_forest

    if models["forest"]["f1"] < 0.55:
        raise ValueError(
            "Random Forest test F1 is below the required 0.55."
        )

    OUT.mkdir(exist_ok=True)

    model_path = OUT / "lapse_model.joblib"

    joblib.dump(selected, model_path)

    loaded_model = joblib.load(model_path)

    original_predictions = selected.predict(X_test)
    reloaded_predictions = loaded_model.predict(X_test)

    assert np.array_equal(
        original_predictions,
        reloaded_predictions
    )

    print("\nModel saved and reloaded successfully.")
    print("Predictions match:", np.array_equal(
        original_predictions, reloaded_predictions
    ))

    # =====================================================
    # 14. PREMIUM CUSTOMER RETENTION LIST
    # Whole-window RFM Premium flag is post-hoc
    # =====================================================

    segments, _, _, _ = rfm_and_segments()

    premium = set(
        segments.loc[
            segments["is_premium"], "customer_id"
        ]
    )

    probabilities = selected.predict_proba(X_test)[
        :, list(selected.classes_).index(1)
    ]

    retention = pd.DataFrame({
        "customer_id": cohort.loc[
            X_test.index, "customer_id"
        ].to_numpy(),
        "lapse_probability": probabilities,
        "actually_lapsed": y_test.to_numpy()
    })

    retention = (
        retention[
            retention["customer_id"].isin(premium)
        ]
        .sort_values(
            "lapse_probability",
            ascending=False
        )
        .head(10)
    )

    retention.to_csv(
        OUT / "retention_list.csv",
        index=False
    )

    print("\n=== PREMIUM CUSTOMER RETENTION LIST ===")
    print(retention.to_string(index=False))

    actual_lapsed_count = int(
        retention["actually_lapsed"].sum()
    )

    print("\nRetention list size:", len(retention))
    print("Actually lapsed customers:", actual_lapsed_count)

    # =====================================================
    # 15. RETURN RESULTS
    # =====================================================

    return {
        "cohort_rows": len(cohort),
        "lapsed": int(y.sum()),
        "baseline": baseline,
        "models": {
            name: {
                key: value
                for key, value in result.items()
                if key != "search"
            }
            for name, result in models.items()
        },
        "frequency_means": {
            "training": train_frequency_mean,
            "full_cohort": full_cohort_frequency_mean
        },
        "feature_importance": feature_importance,
        "logistic_coefficients": coefficient_table,
        "balanced_logistic": balanced_results,
        "retention_list": retention,
        "actual_lapsed_in_retention_list": actual_lapsed_count
    }
