class EntityStateGroup:  # physical/external base state of all entities
    def __init__(self):
        # physical position
        self.p_pos = None
        # physical velocity
        self.p_vel = None

class AgentStateGroup(
    EntityStateGroup
):  # state of agents (including communication and internal/mental state)
    def __init__(self):
        super().__init__()
        # communication utterance
        self.c = None


class ActionGroup:  # action of the agent
    def __init__(self):
        # physical action
        self.u = None
        # communication action
        self.c = None


class EntityGroup:  # properties and state of physical world entity
    def __init__(self):
        # name
        self.name = ""
        # properties:
        self.size = 0.050
        # entity can move / be pushed
        self.movable = False
        # entity collides with others
        self.collide = True
        # material density (affects mass)
        self.density = 25.0
        # color
        self.color = None
        # max speed and accel
        self.max_speed = None
        self.accel = None
        # state
        self.state = EntityStateGroup()
        # mass
        self.initial_mass = 1.0
        # number of entities in the group
        self.n = 0

    @property
    def mass(self):
        return self.initial_mass


class LandmarkGroup(EntityGroup):  # properties of landmark entities
    def __init__(self):
        super().__init__()


class AgentGroup(EntityGroup):  # properties of agent entities
    def __init__(self):
        super().__init__()
        # agents are movable by default
        self.movable = True
        # cannot send communication signals
        self.silent = False
        # cannot observe the world
        self.blind = False
        # physical motor noise amount
        self.u_noise = None
        # communication noise amount
        self.c_noise = None
        # control range
        self.u_range = 1.0
        # state
        self.state = AgentStateGroup()
        # action
        self.action = ActionGroup()
        # script behavior to execute
        self.action_callback = None