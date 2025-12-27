import structlog
from src.core.config import settings

logger = structlog.get_logger()

class Neo4jService:
    def __init__(self):
        self.uri = settings.NEO4J_URI
        self.username = settings.NEO4J_USERNAME
        self.password = settings.NEO4J_PASSWORD
        # self.driver = GraphDatabase.driver(...)
        
    async def query(self, cypher_query):
        pass
