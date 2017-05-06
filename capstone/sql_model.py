from sqlalchemy import Column, Integer, Float, PickleType, ForeignKey, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

Base = declarative_base()
__SESSION__ = None


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
    id = Column(Integer, primary_key=True, unique=True, nullable=False)


class State(Base):
    __tablename__ = 'state'
    id = Column(Integer, primary_key=True)
    target_patch = Column(Integer, ForeignKey("target_patch.id"), nullable=False)
    source = Column(Integer, ForeignKey("source_image.id"))
    x = Column(Integer)
    y = Column(Integer)
    loss = Column(Float)

    def __repr__(self):
        return "<State(source='%s', x='%s', y='%s')>" % (self.source, self.x, self.y)


def get_session(name):
    global __SESSION__
    if not __SESSION__:
        Session = sessionmaker()
        engine = create_engine('sqlite:///progress/%s.sqlite' % name, echo=False)
        Session.configure(bind=engine)
        Base.metadata.create_all(engine)
        __SESSION__ = Session()
    return __SESSION__


def get_target_image(name, k):
    import numpy as np
    session = get_session(name)
    target_image = session.query(TargetImage).filter_by(name=name).first()
    if target_image is None:
        print "Making new project file"
        return make_project_file(name, k, session)
    if k != (np.max(target_image.labels) + 1):
        print "k differs, resaving target image"
        return make_project_file(name, k, session)
    return target_image


def make_project_file(name, k, session):
    import image_segment
    import file_handling

    img = file_handling.get_target_image(name)
    labels = image_segment.segment(img, k)

    target_image = TargetImage(name=name, pixels=img, labels=labels)
    session.merge(target_image)
    session.commit()
    return target_image


if __name__ == "__main__":
    name = "small_butterfly"
    k = 60
    target_image = get_target_image(name, k)

    source_image = SourceImage(id=20)
    state = State(source=source_image.id, translation_x=21, translation_y=22, loss=0.5)
    print state

    target_patch = TargetPatch(visits=[state])
    get_session().add(target_patch)
    get_session().flush()

    print target_patch
    target_image.segments.append(target_patch)
    print target_image.segments

    get_session().merge(target_image)
    get_session().flush()
    get_session().commit()

    t_i = get_target_image(name, k)
    print t_i.segments
