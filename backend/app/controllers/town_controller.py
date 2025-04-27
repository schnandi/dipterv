# app/controllers/town_controller.py
from flask_restx import Namespace, Resource, fields
from flask import request
from app import db
from app.models import Town
import random

# Import your CityGenerator from your generator.py file
# (Assuming you've modified generator.py so that the CityGenerator.generate method returns a dict.)
from generator import CityGenerator

ns = Namespace('towns', description='Town related operations')

# Define models for input and output (for Swagger docs)
town_input_model = ns.model('TownInput', {
    'name': fields.String(description='Optional town name')
})

town_output_model = ns.model('TownOutput', {
    'id': fields.Integer(readOnly=True, description='Unique identifier of a town'),
    'seed': fields.Integer(required=True, description='Random seed used for generation'),
    'name': fields.String(description='Town name (if provided)')
})

town_detail_model = ns.inherit('TownDetail', town_output_model, {
    'data': fields.Raw(description='Full generated town JSON data')
})

@ns.route('/')
class TownList(Resource):
    @ns.doc('list_towns')
    @ns.marshal_with(town_output_model, as_list=True)
    def get(self):
        """
        List all generated towns (id, seed, and optionally name).
        """
        towns = Town.query.all()
        result = [town.to_dict() for town in towns]
        return result

@ns.route('/generate')
class TownGenerate(Resource):
    @ns.doc('generate_town')
    @ns.expect(town_input_model)
    @ns.marshal_with(town_output_model, code=201)
    def post(self):
        """
        Generate a new town with a random seed (or using an optionally provided seed and name).
        """
        payload = request.get_json(force=True, silent=True) or {}

        # Optionally provided name; if not provided, town will only have a seed.
        town_name = payload.get('name')

        # Optionally, a seed can be provided; otherwise, generate one.
        provided_seed = payload.get('seed')
        if provided_seed is None:
            seed = random.randint(0, 2**12 - 1)
            # Ensure the seed is not already in use
            while Town.query.filter_by(seed=seed).first() is not None:
                seed = random.randint(0, 2**12 - 1)
        else:
            seed = int(provided_seed)
            if Town.query.filter_by(seed=seed).first() is not None:
                ns.abort(400, "Seed already exists. Please provide a unique seed or leave it out.")

        # Generate town data using the CityGenerator.
        # (Assuming your CityGenerator.generate method returns the generated JSON data)
        generator = CityGenerator(map_size=(2000, 2000), seed=seed)
        city_data = generator.generate()  # Make sure this returns a dict

        # Create and save the town
        town = Town(seed=seed, name=town_name, data=city_data)
        db.session.add(town)
        db.session.commit()

        return town.to_dict(), 201

@ns.route('/<int:town_id>')
@ns.param('town_id', 'The town identifier')
class TownDetail(Resource):
    @ns.doc('get_town_detail')
    @ns.marshal_with(town_detail_model)
    def get(self, town_id):
        """
        Get the full JSON data for a specific town.
        """
        town = Town.query.get_or_404(town_id, description="Town not found")
        return town.to_full_dict()
