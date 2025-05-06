# app/controllers/simulation_controller.py
from flask_restx import Namespace, Resource, fields
from flask import request
from app import db
from app.models import Simulation, Town
from simulation import simulate_water_network

ns = Namespace('simulations', description='Simulation (time series) operations')

simulation_input = ns.model('SimulationInput', {
    'title':    fields.String(required=True, description='Simulation title'),
    'town_id':  fields.Integer(required=True, description='ID of the town to simulate')
})

simulation_output = ns.model('Simulation', {
    'id':       fields.Integer(readOnly=True, description='Simulation ID'),
    'title':    fields.String(description='Title of the simulation'),
    'town_id':  fields.Integer(description='Linked town ID'),
    'details':  fields.Raw(description='Simulation results')
})

simulation_summary = ns.model('SimulationSummary', {
    'id':      fields.Integer(readOnly=True, description='Simulation ID'),
    'title':   fields.String(description='Title of the simulation'),
    'town_id': fields.Integer(description='Linked town ID'),
})

@ns.route('/')
class SimulationList(Resource):
    @ns.expect(simulation_input)
    @ns.marshal_with(simulation_output, code=201)
    @ns.doc('generate_simulation')
    def post(self):
        """
        Generate time series data for a given town,
        deleting any previous simulation for that town.
        """
        payload = request.get_json()
        town_id = payload['town_id']
        title   = payload['title']

        town = Town.query.get(town_id)
        if not town:
            ns.abort(404, f"Town with id {town_id} not found.")

        # delete any existing simulations for this town
        existing = Simulation.query.filter_by(town_id=town_id).all()
        for sim in existing:
            db.session.delete(sim)
        # flush so that any foreign‐key constraints clear out
        db.session.flush()

        try:
            result = simulate_water_network(town.data)
        except Exception as e:
            ns.abort(500, f"Simulation failed: {str(e)}")

        # create & persist the new simulation
        sim = Simulation(title=title, details=result, town_id=town_id)
        db.session.add(sim)
        db.session.commit()

        return sim, 201

@ns.route('/town/<int:town_id>')
@ns.param('town_id', 'The town identifier')
class SimulationByTown(Resource):
    @ns.doc('get_simulations_for_town')
    @ns.marshal_with(simulation_summary, as_list=True)
    def get(self, town_id):
        """
        Return all simulations for a specific town,
        but only in summary form (no `details` field).
        """
        Town.query.get_or_404(town_id, f"Town {town_id} not found.")
        return Simulation.query.filter_by(town_id=town_id).all()

@ns.route('/<int:simulation_id>')
@ns.param('simulation_id', 'The simulation identifier')
class SimulationResource(Resource):
    @ns.doc('get_simulation')
    @ns.marshal_with(simulation_output)
    def get(self, simulation_id):
        """
        Return details of a specific simulation.
        """
        return Simulation.query.get_or_404(simulation_id)

    @ns.expect(simulation_input)
    @ns.doc('update_simulation')
    def put(self, simulation_id):
        """
        Update simulation metadata only.
        """
        sim = Simulation.query.get_or_404(simulation_id)
        data = request.get_json()

        sim.title = data.get("title", sim.title)
        db.session.commit()
        return {"message": "Simulation updated", "simulation": sim.to_dict()}
