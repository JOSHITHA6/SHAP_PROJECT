if option == "Select Row":

    row = st.number_input("Row Number", 1, len(X_test), 1)

    X_single = X_test[row-1:row]

    # 🔥 ORIGINAL VALUES (no scaling)
    original_row = test_display.iloc[row-1:row].drop(columns=[target])

    _, fig_local = generate_shap_plots(
        model,
        X_test[:100],
        X_single,
        feature_names=feature_names,
        original_row=original_row
    )

    st.pyplot(fig_local)