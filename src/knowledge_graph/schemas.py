from typing import List, Dict, Optional, Literal, Any, Union
from datetime import datetime
from pydantic import BaseModel, Field

# --- Ontology Definitions (Phase 2: Schema Design) ---
# These models define the "Shape of the World" for a specific book series.
# They are used to GENERATE the schema.json files (e.g., 'a_song_of_ice_and_fire_schema.json').
# Think of this as defining the DATABASE SCHEMA (Tables and Columns).
# - "What TYPES of things exist?" (e.g., Characters, Locations)
# - "How CAN they relate?" (e.g., Characters can 'visit' Locations)

class EntityType(BaseModel):
    """Defines a category of entities in the ontology.
    
    Attributes:
        name: The unique name of the entity type (e.g., 'Character').
        description: A clear definition of what this entity type encompasses.
        attributes: A list of common attributes expected for this entity type.
    """
    name: str = Field(..., description="The name of the entity type (e.g., 'Character', 'Location').")
    description: str = Field(..., description="Description of what this entity type represents.")
    attributes: List[str] = Field(default_factory=list, description="Common attributes for this entity type (e.g., 'house', 'wand' for Character).")

class RelationshipType(BaseModel):
    """Defines a directed relationship type between two entity types.
    
    Attributes:
        name: The name of the relationship (e.g., 'knows').
        description: Definition of the relationship semantics.
        source_type: The required entity type for the relationship source.
        target_type: The required entity type for the relationship target.
        properties: List of potential properties for this edge (e.g., 'start_date').
    """
    name: str = Field(..., description="The name of the relationship type (e.g., 'knows', 'sibling_of').")
    description: str = Field(..., description="Description of the relationship.")
    source_type: str = Field(..., description="The entity type that acts as the source.")
    target_type: str = Field(..., description="The entity type that acts as the target.")
    properties: List[str] = Field(default_factory=list, description="Properties associated with this relationship (e.g., 'start_date').")

class SchemaVersion(BaseModel):
    """Metadata tracking the version and evolution of the ontology.
    
    Attributes:
        version: Semantic version of the schema.
        created_at: Timestamp of schema creation/update.
        changelog: List of changes in this version.
        evolution_suggestions: Pending or proposed schema changes.
    """
    version: str = Field(default="1.0.0", description="Semantic version of the schema.")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Timestamp of schema creation/update.")
    changelog: List[str] = Field(default_factory=list, description="List of changes in this version.")
    evolution_suggestions: List[str] = Field(default_factory=list, description="Pending or proposed schema changes.")

class Ontology(BaseModel):
    """The complete schema/ontology for a specific series.
    
    Attributes:
        series_name: The name of the book series this ontology applies to.
        version_info: Schema versioning details.
        entity_types: List of allowed entity types.
        relationship_types: List of allowed relationship types.
        canonical_renaming_rules: Dictionary mapping aliases to canonical names.
    """
    series_name: str = Field(..., description="Name of the book series.")
    version_info: SchemaVersion = Field(default_factory=SchemaVersion, description="Versioning metadata.")
    entity_types: List[EntityType] = Field(..., description="List of valid entity types.")
    relationship_types: List[RelationshipType] = Field(..., description="List of valid relationship types.")
    canonical_renaming_rules: Optional[Dict[str, str]] = Field(default_factory=dict, description="Pre-defined mapping of common aliases to canonical names (optional).")


# --- Extraction Models (Phase 3: Data Ingestion) ---
# These models define the "Shape of the Result" returned by the LLM when processing text.
# They are used to VALIDATE the output from 'extract_entities.py'.
# Think of this as the API RESPONSE FORMAT or Data Rows.
# - "What SPECIFIC things did we find?" (e.g., Found 'Harry Potter', 'Hogwarts')
# - "How ARE they related in this text?" (e.g., 'Harry Potter' is 'at' 'Hogwarts')
# These models often contain extra fields like 'confidence', 'evidence', 'provenance' specific to the extraction job.

class SourceReference(BaseModel):
    """Detailed provenance for an extraction.
    
    Attributes:
        file_name: Name of the source file (e.g., 'chem_section_01.txt').
        chunk_id: Specific chunk identifier if applicable.
    """
    file_name: str = Field(..., description="Name of the source file (e.g., 'chem_section_01.txt').")
    chunk_id: Optional[str] = Field(None, description="Specific chunk identifier if applicable.")
    # In a real system, might include start_char, end_char indices
    
class ExtractionConfidence(BaseModel):
    """Confidence scoring and review flags for extracted items.
    
    Attributes:
        score: Model's confidence score (0.0 to 1.0).
        needs_review: Flag indicating if manual review is recommended.
        review_reason: Reason for flagging for review (e.g., 'ambiguous_reference').
    """
    score: float = Field(..., ge=0.0, le=1.0, description="Model's confidence score (0.0 to 1.0).")
    needs_review: bool = Field(default=False, description="Flag indicating if manual review is recommended.")
    review_reason: Optional[str] = Field(None, description="Reason for flagging for review (e.g., 'ambiguous_reference').")

class EntityInstance(BaseModel):
    """An instance of an entity extracted from text.
    
    Attributes:
        id: Unique canonical ID for the entity (e.g., 'harry_potter').
        type: Entity type from the ontology.
        name: Canonical display name.
        aliases: Aliases found in the text.
        attributes: Extracted attributes.
        description: Brief description/summary from the text.
        mentions: List of places this entity was mentioned.
        confidence: Confidence metadata.
    """
    id: str = Field(..., description="Unique canonical ID for the entity (e.g., 'harry_potter').")
    type: str = Field(..., description="Entity type from the ontology.")
    name: str = Field(..., description="Canonical display name.")
    aliases: List[str] = Field(default_factory=list, description="Aliases found in the text.")
    attributes: Union[Dict[str, Any], str] = Field(default_factory=dict, description="Extracted attributes.")
    description: str = Field(..., description="Brief description/summary from the text.")
    
    # Provenance & Quality
    mentions: List[SourceReference] = Field(default_factory=list, description="List of places this entity was mentioned.")
    confidence: Optional[ExtractionConfidence] = Field(None, description="Confidence metadata.")
    
class RelationshipInstance(BaseModel):
    """An instance of a relationship extracted between two entities.
    
    Attributes:
        source_id: Canonical ID of the source entity.
        target_id: Canonical ID of the target entity.
        type: Relationship type from the ontology.
        description: Contextual description of the relationship.
        evidence: Quote or text snippet proving the relationship.
        properties: Extracted properties (e.g., duration).
        source_ref: Where this relationship was found.
        confidence: Confidence metadata.
    """
    source_id: str = Field(..., description="Canonical ID of the source entity.")
    target_id: str = Field(..., description="Canonical ID of the target entity.")
    type: str = Field(..., description="Relationship type from the ontology.")
    description: str = Field(..., description="Contextual description of the relationship.")
    evidence: str = Field(..., description="Quote or text snippet proving the relationship.")
    properties: Union[Dict[str, Any], str] = Field(default_factory=dict, description="Extracted properties (e.g., duration).")
    
    # Provenance & Quality
    source_ref: Optional[SourceReference] = Field(None, description="Where this relationship was found.")
    confidence: Optional[ExtractionConfidence] = Field(None, description="Confidence metadata.")

class SchemaUpdateProposal(BaseModel):
    """A proposal to update the ontology based on extraction findings.
    
    Attributes:
        proposal_type: Type of update proposed.
        name: Name of the new type or alias.
        description: Reason for the proposal and definition.
        confidence: Confidence that this is a necessary schema change (0.0 to 1.0).
    """
    proposal_type: Literal["new_entity_type", "new_relationship_type", "new_renaming_rule"] = Field(..., description="Type of update proposed.")
    name: str = Field(..., description="Name of the new type or alias.")
    description: str = Field(..., description="Reason for the proposal and definition.")
    confidence: float = Field(..., description="Confidence that this is a necessary schema change (0.0 to 1.0).")

class ExtractionResult(BaseModel):
    """Validates the output of the extraction phase.
    
    This is the top-level object returned by the LLM during extraction.
    """
    entities: List[EntityInstance]
    relationships: List[RelationshipInstance]
    schema_proposals: List[SchemaUpdateProposal] = Field(default_factory=list, description="Proposed updates to the ontology.")
