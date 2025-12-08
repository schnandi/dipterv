# app/controllers/town_controller.py
from flask_restx import Namespace, Resource, fields
from flask import request
from sqlalchemy.orm.attributes import flag_modified

from app import db
from app.models import Town
import random
from app.core.generator.city_generator import CityGenerator

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
    'name': fields.String(description='Town name (if provided)'),
    'data': fields.Raw(description='Full generated town JSON data')
})

leak_model = ns.model('LeakCreate', {
    'fraction': fields.Float(required=False, default=0.5, description='Position along the pipe (0–1)'),
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

    @ns.expect(town_input_model)
    @ns.doc('rename_town')
    def put(self, town_id):
        """Rename an existing town."""
        town = Town.query.get_or_404(town_id)
        payload = request.get_json(force=True) or {}
        town.name = payload.get('name', town.name)
        db.session.commit()
        return town.to_dict()

    @ns.doc('delete_town')
    def delete(self, town_id):
        """Delete a town and all its simulations (including baseline)."""
        town = Town.query.get_or_404(town_id)

        from app.models import Simulation
        sims = Simulation.query.filter_by(town_id=town_id).all()
        for sim in sims:
            db.session.delete(sim)

        db.session.delete(town)
        db.session.commit()
        return {'message': 'Town and all simulations deleted'}



@ns.route('/<int:town_id>/roads/<int:road_id>/leak')
@ns.param('town_id', 'The town identifier')
@ns.param('road_id', 'The road (pipe) identifier')
class RoadLeak(Resource):
    @ns.doc('create_pipe_leak')
    @ns.expect(leak_model, validate=True)
    def post(self, town_id, road_id):
        """
        Mark a pipe as having a leak (store position + rate),
        and invalidate existing simulations for this town.
        """
        payload = request.get_json() or {}
        frac = payload.get('fraction', 0.5)
        rate = payload.get('rate_kg_per_s', 0.01)

        if not (0.0 < frac < 1.0):
            ns.abort(400, "fraction must be between 0 and 1 (exclusive).")

        town = Town.query.get_or_404(town_id)
        data = town.data or {}
        roads = data.get('roads', [])

        for road in roads:
            if road['id'] == road_id:
                x1, y1 = road['start']
                x2, y2 = road['end']
                lx = x1 + (x2 - x1) * frac
                ly = y1 + (y2 - y1) * frac
                road['leak'] = {
                    'coord': [lx, ly],
                    'rate_kg_per_s': rate,
                    'fraction': frac
                }
                break
        else:
            ns.abort(404, f"Road {road_id} not found in town {town_id}")

        # ✅ Invalidate only non-baseline simulations
        from app.models import Simulation  # avoid circular import
        old_sims = Simulation.query.filter(
            Simulation.town_id == town_id,
            Simulation.is_baseline.is_(False)
        ).all()
        for sim in old_sims:
            db.session.delete(sim)

        # ✅ Persist updated town
        flag_modified(town, 'data')
        db.session.commit()

        return {
            'message': 'Leak added successfully. Previous simulations removed.',
            'road_id': road_id,
            'leak': road['leak'],
            'deleted_simulations': len(old_sims)
        }, 200



@ns.route('/<int:town_id>/data')
@ns.param('town_id', 'The town identifier')
class TownDataUpdate(Resource):
    @ns.doc('update_town_data')
    def put(self, town_id):
        """
        Update the full geometry (roads, buildings, junctions, etc.) of a town.
        Automatically deletes any linked simulations so they can be regenerated.
        """
        from app.models import Simulation  # import here to avoid circular refs

        town = Town.query.get_or_404(town_id, description="Town not found")
        payload = request.get_json(force=True) or {}

        if not isinstance(payload, dict):
            ns.abort(400, "Invalid data: expected JSON object with town data.")

        # Update the town data
        data = town.data or {}

        for key in ("roads", "buildings", "junctions"):
            if key in payload:
                data[key] = payload[key]

        town.data = data
        flag_modified(town, "data")

        # Delete any existing simulations for this town
        old_sims = Simulation.query.filter_by(town_id=town_id).all()
        for sim in old_sims:
            db.session.delete(sim)

        db.session.commit()

        return {
            "message": "Town data updated successfully. Existing simulations removed.",
            "deleted_simulations": len(old_sims)
        }, 200