# app/__init__.py
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_restx import Api


db = SQLAlchemy()

def create_app():
    app = Flask(__name__)
    app.config.from_object('app.config.Config')

    db.init_app(app)

    # Create the API instance and configure Swagger documentation.
    api = Api(app,
              version='0.1.0',
              title='Town & Simulation API',
              description='An API for town generation and simulation data',
              doc='/docs')  # Swagger UI available at /docs

    # Import and add namespaces (these will be your controllers).
    from app.controllers import town_controller, simulation_controller
    api.add_namespace(town_controller.ns)
    api.add_namespace(simulation_controller.ns)

    return app
