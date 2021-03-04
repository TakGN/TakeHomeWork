import json

from sqlalchemy import Column, Integer, String, Text, Boolean, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, validates

import settings

Base = declarative_base()


class Connection:
    @staticmethod
    def connect():
        """
        Establish a connection with the database.

        Returns:
            db_session: a database session.

        """
        database = settings.DATABASE['PATH']
        engine = create_engine(database)
        Base.metadata.create_all(engine)
        Base.metadata.bind = engine
        db_session = sessionmaker(bind=engine)
        db_session = db_session()
        db_session.execute('pragma foreign_keys=on')
        return db_session


class TrainModel(Base):
    __tablename__ = 'models'
    id = Column(Integer, primary_key=True)
    name = Column(String, unique=True, nullable=False)
    accuracy = Column(Integer, nullable=False)
    train_date = Column(DateTime, nullable=False)
    serving = Column(Boolean, nullable=False)
    model_params = Column(Text, nullable=False)

    def dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'accuracy': self.accuracy,
            'train_date': self.train_date,
            'serving': self.serving,
            'model_params': json.loads(self.model_params)
        }

    @classmethod
    def add(cls, **kwargs):
        """
        Adds a new model in the database.
        Args:
            **kwargs:
             the collection of arguments necessary to create the desired model.

        Returns:
            new_id: the newly added model id.

        """
        session = Connection.connect()
        new_element = cls(**kwargs)
        session.add(new_element)
        session.commit()
        new_id = new_element.id
        new_item = session.query(cls).filter_by(id=new_id).one()
        session.close()
        return new_item

    @classmethod
    def get(cls, model_id):
        """
        Gets a specific model from the database.

        Args:
            model_id: the to-be-queried model id.

        Returns:
            record: the desired model details.

        """
        session = Connection.connect()
        q = session.query(cls)
        q = q.filter_by(id=model_id)
        record = q.one()
        session.close()
        return record

    @classmethod
    def query(cls):
        """
        Queries the database for all the models.

        Returns:
            record: A list of all the models' instances
                    currently present in the database.
        """
        session = Connection.connect()
        record = session.query(cls).all()
        session.close()
        return record

    @classmethod
    def edit(cls, id, **kwargs):
        """
        Updates a certain item in the database.

        Args:
            **kwargs:
               the collection of arguments necessary to update the desired item.
        """
        session = Connection.connect()
        q = session.query(cls)
        q = q.filter_by(id=id).one()
        for key, value in kwargs.items():
            setattr(q, key, value)
        session.commit()
        record = session.query(cls).filter_by(id=id).one()
        session.close()
        return record



