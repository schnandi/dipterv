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
        """Delete a town."""
        town = Town.query.get_or_404(town_id)
        db.session.delete(town)
        db.session.commit()
        return {'message': 'Town deleted'}



@ns.route('/<int:town_id>/roads/<int:road_id>/leak')
@ns.param('town_id', 'The town identifier')
@ns.param('road_id', 'The road (pipe) identifier to split')
class RoadLeak(Resource):
    @ns.doc('create_pipe_leak')
    @ns.expect(leak_model, validate=True)
    def post(self, town_id, road_id):
        """
        Split a pipe at the given fraction and mark the split-point as a leak junction.
        """
        payload = request.get_json() or {}
        frac = payload.get('fraction', 0.5)
        if not (0.0 < frac < 1.0):
            ns.abort(400, "fraction must be between 0 and 1 (exclusive).")

        town = Town.query.get_or_404(town_id, description="Town not found")
        data = town.data or {}
        roads = data.setdefault('roads', [])
        leaks = data.setdefault('leaks', [])

        # find and remove original road
        for i, r in enumerate(roads):
            if r.get('id') == road_id:
                original = roads.pop(i)
                break
        else:
            ns.abort(404, f"Road with id {road_id} not found in town {town_id}")

        x1, y1 = original['start']
        x2, y2 = original['end']
        # compute leak junction
        lx = x1 + (x2 - x1) * frac
        ly = y1 + (y2 - y1) * frac
        leak_coord = [lx, ly]

        # generate new unique road IDs
        existing_ids = {r['id'] for r in roads}
        max_id = max(existing_ids) if existing_ids else road_id
        new_id1, new_id2 = max_id + 1, max_id + 2

        # split into two pipes
        meta = {k: original[k] for k in ('pipe_type','age') if k in original}
        seg1 = {
            'id': new_id1,
            'start': original['start'],
            'end': leak_coord,
            **meta
        }
        seg2 = {
            'id': new_id2,
            'start': leak_coord,
            'end': original['end'],
            **meta
        }
        roads.extend([seg1, seg2])

        # record leak info for downstream processing
        leak_entry = {
            'original_road_id': road_id,
            'leak_junction': leak_coord,
            'split_road_ids': [new_id1, new_id2]
        }
        leaks.append(leak_entry)

        # save
        town.data = data
        flag_modified(town, 'data')
        db.session.commit()

        return {
            'message': 'Pipe split and leak junction created.',
            'leak': leak_entry,
            'new_pipes': [seg1, seg2]
        }, 200