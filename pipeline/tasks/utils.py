from airflow.models import Variable

def get_execution_date_str(context) -> str:
    """
    Get the execution date string for this DAG run.

    Priority:
        1. PIPELINE_TEST_DATE variable (for demos and testing)
        2. logical_date from context (Airflow 2.8+)
        3. execution_date from context (fallback)

    To run in demo mode:
        airflow variables set PIPELINE_TEST_DATE "2025-12-15"

    To run in production mode:
        airflow variables delete PIPELINE_TEST_DATE
    """
    test_date = Variable.get('PIPELINE_TEST_DATE', default_var=None)
    if test_date:
        print(f"[utils] Using test date: {test_date}")
        return test_date

    logical_date = context.get('logical_date') or context.get('execution_date')
    return logical_date.strftime('%Y-%m-%d')