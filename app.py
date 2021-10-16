from flask_restplus import Api, Resource
from flask import Flask

from webargs.flaskparser import use_args
from webargs import fields

from tempNN import predict_temperature
from humNN import predict_humidity

app = Flask(__name__)
api = Api(app)


temp_input_size = 24
hum_input_size = 36
elem_per_row = 7


def validate_entry(num_rows: int, rows: list[list[float]]) -> tuple[bool, str]  :
    actual_len = len(rows)
    if actual_len != num_rows:
        return False, f"expected #total-rows: {num_rows}, got {actual_len}"
    
    for idx, row in enumerate(rows):
        row_len = len(row)
        if row_len != elem_per_row:
            return False, f"expected row-#{idx}-length: {elem_per_row}, got {row_len}"
    
    return True, ""


def farenheit_to_celsius(temp):
    return (temp - 32) / 1.8


@api.route("/api/predict/temperature")
class TemperaturePredicter(Resource):
    @use_args({
        "last_data": fields.List(fields.List(fields.Float), required=True)
    })
    def post(self, args):
        valid, msg = validate_entry(temp_input_size, args["last_data"])
        if not valid:
            return {"error": "input data has invalid dimensions; " + msg}, 422
        
        data = args["last_data"]
        norm_output = farenheit_to_celsius(predict_temperature(data))
        return {"temperature": norm_output}, 200


@api.route("/api/predict/humidity")
class HumidityPredicter(Resource):
    @use_args({
        "last_data": fields.List(fields.List(fields.Float), required=True)
    })
    def post(self, args):
        valid, msg = validate_entry(hum_input_size, args["last_data"])
        if not valid:
            return {"error": "input data has invalid dimensions;" + msg}, 422
        
        data = args["last_data"]
        norm_output = predict_temperature(data)
        return {"humidity": norm_output}, 200


if __name__ == "__main__":
    app.run()
