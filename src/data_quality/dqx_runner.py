from databricks.labs.dqx.engine import DQEngine
from databricks.sdk import WorkspaceClient
from pyspark.sql import functions as F
import yaml
from pathlib import Path

CONTRACTS_DIR = Path(__file__).resolve().parents[1] / "contracts"

AUDIT_RESULTS_TABLE = "medallion_catalog.audit.data_quality_results"
AUDIT_ERRORS_TABLE = "medallion_catalog.audit.data_quality_errors"


def get_contract_from_table_name(table_name: str) -> str:
    """Return the raw YAML contract for a given catalog.schema.table name."""
    _, schema, table = table_name.split(".")
    contract_path = CONTRACTS_DIR / schema / f"{table}.yml"

    if not contract_path.exists():
        raise FileNotFoundError(
            f"No data quality contract found for table '{table_name}' "
            f"(expected at {contract_path})"
        )

    return contract_path.read_text()


def run_dqx(table_name: str, contract_text: str):
    ws = WorkspaceClient()
    engine = DQEngine(ws)

    contract = yaml.safe_load(contract_text)
    checks = contract["checks"]

    status_checks = engine.validate_checks(checks)
    if status_checks.has_errors:
        raise ValueError(f"Invalid contract for {table_name}: {status_checks.errors}")

    df = spark.table(table_name)
    df_results = engine.apply_checks_by_metadata(df, checks)

    return df_results


def build_quality_summary_df(df_results, table_name: str):
    """One-row summary of a DQX run: overall result, row counts, and error/warning rates."""
    rows_analyzed = df_results.count()

    agg = df_results.select(
        F.when(F.size(F.coalesce("_errors", F.array())) > 0, 1).otherwise(0).alias("has_error"),
        F.when(F.size(F.coalesce("_warnings", F.array())) > 0, 1).otherwise(0).alias("has_warning"),
    ).agg(
        F.sum("has_error").alias("rows_with_error"),
        F.sum("has_warning").alias("rows_with_warning"),
    ).first()

    rows_with_error = agg["rows_with_error"] or 0
    rows_with_warning = agg["rows_with_warning"] or 0

    if rows_with_error > 0:
        overall_result = "fail"
    elif rows_with_warning > 0:
        overall_result = "pass_with_warnings"
    else:
        overall_result = "pass"

    pct_error = (rows_with_error / rows_analyzed * 100) if rows_analyzed else 0.0
    pct_warning = (rows_with_warning / rows_analyzed * 100) if rows_analyzed else 0.0

    return df_results.sparkSession.createDataFrame([{
        "table_name": table_name,
        "overall_result": overall_result,
        "rows_analyzed": rows_analyzed,
        "pct_rows_with_error": pct_error,
        "pct_rows_with_warning": pct_warning,
        "amount_rows_with_error": rows_with_error,
        "amount_rows_with_warning": rows_with_warning,
    }]).withColumn("run_date", F.current_date())


def build_quality_errors_df(df_results, table_name: str):
    """One row per distinct failing/warning check, with the count of rows it affected."""
    errors = (
        df_results.select(F.explode("_errors").alias("issue"))
        .withColumn("type", F.lit("error"))
    )
    warnings = (
        df_results.select(F.explode("_warnings").alias("issue"))
        .withColumn("type", F.lit("warn"))
    )

    exploded = errors.unionByName(warnings)

    return (
        exploded
        .groupBy("type", F.col("issue.name").alias("type_description"))
        .agg(F.count(F.lit(1)).alias("count_of_evidences"))
        .withColumn("table_name", F.lit(table_name))
        .withColumn("run_date", F.current_date())
        .select("table_name", "run_date", "type", "type_description", "count_of_evidences")
    )


def write_append(df, target_table: str) -> None:
    df.write.mode("append").saveAsTable(target_table)


def main(table_name: str) -> None:
    contract_text = get_contract_from_table_name(table_name)
    df_results = run_dqx(table_name, contract_text)

    summary_df = build_quality_summary_df(df_results, table_name)
    write_append(summary_df, AUDIT_RESULTS_TABLE)

    errors_df = build_quality_errors_df(df_results, table_name)
    write_append(errors_df, AUDIT_ERRORS_TABLE)
