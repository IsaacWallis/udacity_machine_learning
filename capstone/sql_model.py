from sqlalchemy import Column, Integer, Float, PickleType, ForeignKey, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

Base = declarative_base()
Session = sessionmaker()


class TargetImage(Base):
    __tablename__ = 'target_image'
    name = Column(String, primary_key=True)
    pixels = Column(PickleType, nullable=False)
    labels = Column(PickleType, nullable=False)
    segments = relationship("TargetPatch")


class TargetPatch(Base):
    __tablename__ = 'target_patch'
    id = Column(Integer, primary_key=True)
    parent_name = Column(String, ForeignKey("target_image.name"))
    visits = relationship("State")

    def __repr__(self):
        return "<Target Patch(id='%s')>" % self.id


class SourceImage(Base):
    __tablename__ = 'source_image'
    id = Column(Integer, primary_key=True)


class State(Base):
    __tablename__ = 'state'
    id = Column(Integer, primary_key=True)
    searching_patch_name = Column(Integer, ForeignKey("target_patch.id"))
    source = ForeignKey("source_image.id")
    translation_x = Column(Integer)
    translation_y = Column(Integer)
    loss = Column(Float)

    def __repr__(self):
        return "<State(source='%s', x='%s', y='%s')>" % (self.id, self.translation_x, self.translation_y)


def set_sql_path(name):
    engine = create_engine('sqlite:///progress/%s.sqlite' % name, echo=True)
    Session.configure(bind=engine)
    Base.metadata.create_all(engine)
    return Session()


def get_target_image(name):
    session = Session()
    target_image = session.query(TargetImage).filter_by(name=name).one()
    return target_image


if __name__ == "__main__":
    import image_segment
    import os
    from scipy import ndimage
    import numpy as np

    name = "small_butterfly"
    session = set_sql_path(name)
    target_image = session.query(TargetImage).filter_by(name=name).first()

    k = 50
    if k != (np.max(target_image.labels) + 1):
        print "resaving target image"
        image_dir = "./images"

        path = os.path.join(image_dir, name + ".jpg")
        img = ndimage.imread(path)
        labels = image_segment.segment(img, k)

        target_image = TargetImage(name=name, pixels=img, labels=labels)
        session.merge(target_image)
        session.commit()




    print session
