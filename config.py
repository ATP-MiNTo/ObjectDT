from dotenv import load_dotenv
import os

load_dotenv()
API_KEY     = os.getenv("ROBOFLOW_API_KEY")
Space_KEY   = os.getenv("ROBOFLOW_Space_KEY")
Project_KEY     = os.getenv("ROBOFLOW_Project_KEY")
Version_KEY     = os.getenv("ROBOFLOW_Version_KEY")