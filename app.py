from flask import Flask, request, jsonify
from joblib import load
from pathlib import Path
from models.predicion import predecir_paciente

# 🔹 Ruta base donde está este archivo (app.py)
BASE_DIR = Path(__file__).resolve().parent

# 🔹 Cargar modelo y scaler usando rutas compatibles con Linux/Windows
model_rl = load(BASE_DIR / "models" / "modelo_regresion_logistica.pkl")
scaler = load(BASE_DIR / "models" / "scaler.pkl")

app = Flask(__name__)

@app.route("/api/hepatitis", methods=["POST"])
def calcular_prediccion_endpoint():
    # 1️⃣ Verificar que se envíe JSON
    if not request.is_json:
        return jsonify({"error": "El contenido debe ser JSON"}), 400

    data = request.get_json()

    print("sadasasdsass")

    # 2️⃣ Todos los campos requeridos
    campos_requeridos = [
        "Age", "Sex_encoded", "Estado_Civil_encoded", "Ciudad_encoded", "Steroid",
        "Antivirals", "Fatigue", "Malaise", "Anorexia", "Liver_Big", "Liver_Firm",
        "Spleen_Palpable", "Spiders", "Ascites", "Varices", "Bilirubin",
        "Alk_Phosphate", "Sgot", "Albumin", "Protime", "Histology"
    ]

    # 3️⃣ Validar que todos los campos existan
    for campo in campos_requeridos:
        if campo not in data:
            return jsonify({"error": f"Falta el campo '{campo}' en el JSON"}), 400

    try:
        # 4️⃣ Convertir todos los campos a float
        valores = [float(data[campo]) for campo in campos_requeridos]

        # 5️⃣ Validación simple: todos los valores >= 0
        if any(v < 0 for v in valores):
            return jsonify({"error": "Todos los valores deben ser mayores o iguales a cero"}), 400

        # 6️⃣ Llamar a la función de predicción con modelo y scaler
        try:
            resultado_modelo = predecir_paciente(model_rl, scaler, valores)
        except Exception as e:
            return jsonify({"error_interno": str(e)}), 500

        # 7️⃣ Devolver resultado en JSON
        return jsonify({
            "entrada": data,
            "resultado_modelo": resultado_modelo
        }), 200

    except ValueError:
        return jsonify({"error": "Todos los valores deben ser numéricos"}), 400


# Endpoint de prueba
@app.route("/api/hepatitis/ejemplo", methods=["GET"])
def ejemplo():
    ejemplo_data = {
        "modelo": "Modelo de Hepatitis",
        "random_state": 42,
        "max_iter": 1000,
        "metricas_train": {"accuracy": 1.0, "precision": 1.0, "recall": 1.0, "f1": 1.0},
        "metricas_test": {"accuracy": 1.0, "precision": 1.0, "recall": 1.0, "f1": 1.0},
        "n_features": 21,
        "features": [
            "Age", "Sex_encoded", "Estado_Civil_encoded", "Ciudad_encoded", "Steroid",
            "Antivirals", "Fatigue", "Malaise", "Anorexia", "Liver_Big", "Liver_Firm",
            "Spleen_Palpable", "Spiders", "Ascites", "Varices", "Bilirubin",
            "Alk_Phosphate", "Sgot", "Albumin", "Protime", "Histology"
        ]
    }
    return jsonify(ejemplo_data), 200


if __name__ == "__main__":
    app.run(debug=True)
