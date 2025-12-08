# app/__init__.py
from flask import Flask
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from flask_restx import Api


db = SQLAlchemy()

def create_app():
    app = Flask(__name__)
    app.config.from_object('app.config.Config')

    CORS(app, resources={r"/*": {"origins": "http://localhost:3000"}})

    db.init_app(app)

    # Create the API instance and configure Swagger documentation.
    api = Api(app,
              version='0.1.0',
              title='Town & Simulation API',
              description='An API for town generation and simulation data',
              doc='/docs')  # Swagger UI available at /docs

    # Import and add namespaces (these will be your controllers).
    from app.controllers import town_controller, simulation_controller, burst_risk_controller, leak_risk_controller
    api.add_namespace(town_controller.ns)
    api.add_namespace(simulation_controller.ns)
    api.add_namespace(burst_risk_controller.ns)
    api.add_namespace(leak_risk_controller.ns)

    return app
