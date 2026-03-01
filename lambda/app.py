from run_model import load_model
import json
import numpy as np
print("start")
model = load_model()  # load once per container
print("done loading")
def handler(event, context):
    try:
        # Safe parsing
        body_str = event.get("body") or "{}"
        body = json.loads(body_str)
        text = body.get("text", "")
        if not isinstance(text, str):
            text = str(text)
        print("Got here!")
        # Run model
        output = model.gen(text)
        print("dpne run!")
        return {
            "statusCode": 200,
            "headers": {
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*"
            },
            "body": json.dumps({"output": output})
        }
    except Exception as e:
        # Catch everything and return a JSON error
        import traceback
        tb = traceback.format_exc()
        print(tb)
        return {
            "statusCode": 500,
            "headers": {
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*"
            },
            "body": json.dumps({"error": str(e), "traceback": tb})
        }
