# app/models.py
from app import db

class Town(db.Model):
    __tablename__ = 'towns'
    id = db.Column(db.Integer, primary_key=True)
    seed = db.Column(db.Integer, unique=True, nullable=False)
    name = db.Column(db.String(255), nullable=True)  # Optional town name
    data = db.Column(db.JSON, nullable=False)  # Full JSON data for the town

    simulations = db.relationship(
        'Simulation',
        back_populates='town',
        cascade='all, delete-orphan'
    )

    def to_dict(self):
        return {
            "id": self.id,
            "seed": self.seed,
            "name": self.name,
            # For list endpoints you might not want to send full JSON,
            # so you can omit `data` here (or include it in detail views)
        }

    def to_full_dict(self):
        """Return a full representation including the town data."""
        d = self.to_dict()
        d["data"] = self.data
        return d


class Simulation(db.Model):
    __tablename__ = 'simulations'
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(255), nullable=False)
    details = db.Column(db.JSON, nullable=True)  # JSON result from simulation
    town_id = db.Column(db.Integer, db.ForeignKey('towns.id', ondelete='CASCADE'), nullable=False)
    town = db.relationship('Town', back_populates='simulations')

    is_baseline = db.Column(db.Boolean, default=False)
    baseline_id = db.Column(db.Integer, db.ForeignKey('simulations.id'), nullable=True)

    def to_dict(self):
        return {
            "id": self.id,
            "title": self.title,
            "town_id": self.town_id,
            "details": self.details,  # or summary only if it's large
        }
