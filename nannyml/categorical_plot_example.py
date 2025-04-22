# https://community.plotly.com/t/plotly-subplots-with-individual-legends/1754/25

import nannyml as nml


def main():
    reference_df, analysis_df, _ = nml.load_synthetic_car_loan_dataset()

    column_names = [
        "car_value",
        "salary_range",
        "debt_to_income_ratio",
        "loan_length",
        "repaid_loan_on_prev_car",
        "size_of_downpayment",
        "driver_tenure",
        "y_pred_proba",
        "y_pred",
    ]

    calc = nml.UnivariateDriftCalculator(
        column_names=column_names,
        treat_as_categorical=["y_pred"],
        timestamp_column_name="timestamp",
        categorical_methods=["jensen_shannon"],
    )

    calc.fit(reference_df)

    result = calc.calculate(analysis_df)

    figure = result.filter(
        column_names=result.categorical_column_names,
        methods=["jensen_shannon"],
    ).plot(kind="distribution")

    figure.show()
