from shiny import App, ui, render, reactive
import pandas as pd
import torch
import os
from HINT.dataloader import csv_three_feature_2_dataloader_for_prediction  
from HINT.model import HINTModel
from HINT.molecule_encode import MPNN
from HINT.icdcode_encode import GRAM, build_icdcode2ancestor_dict
from HINT.protocol_encode import Protocol_Embedding

# Model paths
MODEL_PATHS = {
    "Phase I": "save_model/phase_I.ckpt",
    "Phase II": "save_model/phase_II.ckpt",
    "Phase III": "save_model/phase_III.ckpt",
}

PHASE_OPTIONS = ["Phase I", "Phase II", "Phase III", "All Phases"]

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 🎨 UI with Light/Dark Mode Toggle
app_ui = ui.page_sidebar(
    ui.sidebar(
        ui.h2("📊 Clinical Trial Phase Prediction", class_="title-text"),  
        ui.input_select("phase", "📌 Select Clinical Trial Phase", PHASE_OPTIONS, selected="Phase I"),
        ui.input_file("file", "📂 Upload CSV File", accept=[".csv"], multiple=False),
        ui.input_action_button("predict", "🚀 Predict", class_="btn btn-success w-100 mt-3"),
        ui.input_action_button("toggle_mode", "🌙 Dark Mode", class_="btn btn-secondary w-100 mt-2", onclick="toggleTheme()"),  
        ui.hr(),
        ui.tags.div(ui.output_text("status_message"), class_="status-message text-center mt-3"),  
        width=450
    ),
    ui.layout_columns(
        ui.card(
            ui.h3("📈 Prediction Results", class_="text-dark"),
            ui.output_data_frame("prediction_table"),  
            ui.br(),
            ui.download_button("download", "📥 Download Predictions", class_="btn download-btn"),  
            class_="p-4 bg-light rounded shadow-lg"
        )
    ),
    # 🎨 Dark Mode Fix
    ui.tags.style("""
        body.light-mode {
            background-color: #f4f4f4;
            color: #000000;
        }
        body.dark-mode {
            background-color: #1e1e1e;
            color: #ffffff;
        }
        .sidebar {
            background-color: #2c3e50;
            padding: 20px;
            height: 100vh;
        }
        .title-text, .status-message {
            color: #000000 !important;
            font-weight: bold;
        }
        .btn-success {
            background-color: #28a745;
            border: none;
        }
        .download-btn {  
            background-color: #5bc0de !important;
            color: #000000 !important;
            font-weight: bold;
        }
        .dark-mode .sidebar {
            background-color: #111111 !important;
        }
        .dark-mode .title-text, .dark-mode .status-message {
            color: #ffffff !important;
        }
        .dark-mode .download-btn {
            background-color: #3498db !important;
            color: #ffffff !important;
        }
    """),
    # ✅ JavaScript for Dark Mode Toggle
    ui.tags.script("""
        function toggleTheme() {
            var body = document.body;
            var button = document.querySelector("button[onclick='toggleTheme()']");
            if (body.classList.contains("dark-mode")) {
                body.classList.remove("dark-mode");
                body.classList.add("light-mode");
                button.innerText = "🌙 Dark Mode";
            } else {
                body.classList.remove("light-mode");
                body.classList.add("dark-mode");
                button.innerText = "☀️ Light Mode";
            }
        }
    """)
)


# Server Logic
def server(input, output, session):
    
    predictions_df = reactive.Value(pd.DataFrame())

    def run_prediction(selected_phase, file_path):
        output_file = "predictions_no_labels.xlsx"

        if selected_phase == "All Phases":
            all_data = []
            writer = pd.ExcelWriter(output_file, engine="xlsxwriter")

            for phase in ["Phase I", "Phase II", "Phase III"]:
                model_path = MODEL_PATHS[phase]

                if not os.path.exists(model_path):
                    continue  

                test_loader = csv_three_feature_2_dataloader_for_prediction(file_path, shuffle=False, batch_size=32)
                model = torch.load(model_path, map_location=device)
                model.eval()

                # ✅ Using ongoing_test() (No labels)
                nctid_all, predict_all = model.ongoing_test(test_loader)

                df_phase = pd.DataFrame({
                    "NCT_ID": nctid_all,
                    "Phase": phase,
                    "Prediction_Probability": predict_all
                })
                df_phase.to_excel(writer, sheet_name=phase, index=False)
                all_data.append(df_phase)

            writer.close()
            return pd.concat(all_data, ignore_index=True), f"✅ Predictions saved to {output_file}"

        elif selected_phase in MODEL_PATHS:
            model_path = MODEL_PATHS[selected_phase]
            test_loader = csv_three_feature_2_dataloader_for_prediction(file_path, shuffle=False, batch_size=32)

            if not os.path.exists(model_path):
                return pd.DataFrame(), f"Checkpoint missing for {selected_phase}"

            model = torch.load(model_path, map_location=device)
            model.eval()

            # ✅ Using ongoing_test() (No labels)
            nctid_all, predict_all = model.ongoing_test(test_loader)

            df = pd.DataFrame({
                "NCT_ID": nctid_all,
                "Phase": selected_phase,  
                "Prediction_Probability": predict_all
            })
            df.to_excel(output_file, index=False)

            return df, f"✅ Predictions saved to {output_file}"

        return pd.DataFrame(), "⚠️ No valid model found!"

    @render.data_frame
    def prediction_table():
        return predictions_df.get()

    @render.text
    def status_message():
        return "Click 'Predict' to start" if predictions_df.get().empty else "✅ Prediction completed successfully!"

    @output
    @render.download
    def download():
        return "predictions_no_labels.xlsx"

    @reactive.effect
    @reactive.event(input.predict)
    def predict():
        file = input.file()
        selected_phase = input.phase()

        if not file:
            predictions_df.set(pd.DataFrame())  
            return

        file_path = file[0]["datapath"]
        df, message = run_prediction(selected_phase, file_path)

        predictions_df.set(df)
        print(message)  


# Run the app
app = App(app_ui, server)

if __name__ == "__main__":
    app.run()
