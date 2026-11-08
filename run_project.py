"""Run all project stages in sequence. Use this after installing requirements."""
print("🚀 RUN_PROJECT.PY STARTED")
import profile

from src.data_pipeline import audit,clean_data,validate
from src.sql_analysis import run_sql_analysis
from src.eda_analysis import run_eda_analysis
from src.customer_models import rfm_and_segments,build_lapse_model
from src.triage import run_pipeline, ZERO, FEW
from src.policy_rag import PolicyRAG,evaluate
def main():
    print("\n=== PART A: AUDIT / CLEAN / VALIDATE ===")
    audit(); clean_data(); validate()
    print("\n=== SQL ANALYSIS ===")
    sql_results = run_sql_analysis()
    print("SQL Analysis Results:")
    print(sql_results)
    print("\n=== EXPLORATORY DATA ANALYSIS ===")
    eda_results = run_eda_analysis(sql_results["revenue"])
    print("\n=== PART B: SEGMENTS ===")
    
    print("\n=== PART B: RFM AND SEGMENTS ===")

    rfm, diagnostics, profile, pca = rfm_and_segments(k=4)

    print("\nRFM sample:")
    print(rfm.head().to_string(index=False))

    print("\nK-means diagnostics:")
    print(diagnostics.to_string(index=False))

    print("\nSegment profile:")
    print(profile.to_string(index=False))

    print("\nPCA explained variance:", pca)
    print("Total PCA explained variance:", pca.sum())


    print("\n=== PART B: LAPSE MODEL ===")

    lapse_summary = build_lapse_model()

    print("\nCohort rows:", lapse_summary["cohort_rows"])
    print("Lapsed customers:", lapse_summary["lapsed"])

    print("\nBaseline metrics:")
    print(lapse_summary["baseline"])

    print("\nModel comparison:")
    for model_name, results in lapse_summary["models"].items():
        print(f"\n{model_name.upper()}")
        print(results)

    print("\nTraining vs full-cohort frequency mean:")
    print(lapse_summary["frequency_means"])

    print("\nTop 5 Random Forest feature importances:")
    print(lapse_summary["feature_importance"].head(5))

    print("\nLogistic Regression coefficients:")
    print(lapse_summary["logistic_coefficients"].head(10))

    print("\nOriginal vs oversampled Logistic Regression:")
    print(lapse_summary["balanced_logistic"])

    print("\nRetention list:")
    print(lapse_summary["retention_list"])

    print(
    "\nActual lapses in retention list:",
    lapse_summary["actual_lapsed_in_retention_list"]
    )
    print("\n=== PART C: TRIAGE ===")
    
    print("\n=== PART C: TRIAGE ===")

# 1. ZERO-SHOT RUN
    print("\n--- ZERO-SHOT EVALUATION ---")

    zero_results, zero_summary = run_pipeline(
    prompt=ZERO,
    token_budget=100000
    )

    print("Classified tickets:", zero_summary["classified_tickets"])
    print("Accuracy:", round(zero_summary["accuracy"] * 100, 2), "%")
    print("Total tokens:", zero_summary["total_tokens"])
    print(
    "Tokens per classified ticket:",
    round(zero_summary["tokens_per_classified_ticket"], 2)
    )
    print("Status counts:", zero_summary["status_counts"])

    print("\nZero-shot confusion matrix:")
    print(zero_summary["confusion_matrix"])


# 2. FEW-SHOT RUN
    print("\n--- FEW-SHOT EVALUATION ---")

    few_results, few_summary = run_pipeline(
    prompt=FEW,
    token_budget=100000
    )

    print("Classified tickets:", few_summary["classified_tickets"])
    print("Accuracy:", round(few_summary["accuracy"] * 100, 2), "%")
    print("Total tokens:", few_summary["total_tokens"])
    print(
    "Tokens per classified ticket:",
    round(few_summary["tokens_per_classified_ticket"], 2)
    )
    print("Status counts:", few_summary["status_counts"])


# 3. TOKEN BUDGET DEMONSTRATION
    print("\n--- TOKEN BUDGET DEMONSTRATION ---")

    budget_results, budget_summary = run_pipeline(
    prompt=ZERO,
    token_budget=6000,
    output_file="triage_budget_demo.csv"
    )

    print("Token budget:", 6000)
    print("Tokens consumed:", budget_summary["total_tokens"])
    print(
    "Budget exhausted tickets:",
    budget_summary["status_counts"].get("budget_exhausted", 0)
    )


# 4. FINAL FEW-SHOT RUN
    print("\n--- FINAL TRIAGE OUTPUT ---")

    final_results, final_summary = run_pipeline(
    prompt=FEW,
    token_budget=100000,
    output_file="triage_results.csv"
    )

    print("Final status counts:", final_summary["status_counts"])
    print("Final total tokens:", final_summary["total_tokens"])
    print("Final accuracy:", round(final_summary["accuracy"] * 100, 2), "%")

    print("\nFinal triage preview:")
    print(final_results.head(10).to_string(index=False))

    print("\nPart C execution completed.")

    print("\n=== PART D: POLICY RAG ===")
    rag=PolicyRAG(); print("chunks:",len(rag.chunks)); print(evaluate(rag).to_string(index=False))
    print("\nProject pipeline finished. Review outputs/ and data/instamart.db.")
if __name__=="__main__":
    main()
