from flask import Flask,request,Response,jsonify
from long_text_inferencing import tokenize_long_text,model_inference_long_text


app = Flask(__name__)

@app.route("/",methods = ["POST","GET"])
def raghav():
    return jsonify(message = "Hello world")

@app.route("/predict", methods=["POST"])
def predict_entity():
    if request.method == "POST":
        if "text" in request.form:
            text = request.form["text"]
        elif request.json and "text" in request.json:
            text = request.json["text"]
        else:
            return jsonify(error="No 'text' field provided"), 400

        input_dict = tokenize_long_text(text)
        entity = model_inference_long_text(input_dict, text)
        return Response(f"Entities: {entity}")

    
    
    
    
if __name__ == "__main__":
    app.run(host = '0.0.0.0', port = 5000, debug=True)
        
        