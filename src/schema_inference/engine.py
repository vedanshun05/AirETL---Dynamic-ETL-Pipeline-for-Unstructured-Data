"""
AI-powered Schema Inference Engine for Dynamic ETL Pipeline.
Automatically detects and adapts to evolving data structures.
"""

from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import json
import numpy as np
import pandas as pd
from collections import defaultdict, Counter
import hashlib
import re

from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sentence_transformers import SentenceTransformer
import tensorflow as tf
from loguru import logger


class DataType(Enum):
    """Enumeration of supported data types."""
    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    DATETIME = "datetime"
    DATE = "date"
    EMAIL = "email"
    URL = "url"
    JSON = "json"
    ARRAY = "array"
    OBJECT = "object"
    NULL = "null"
    UNKNOWN = "unknown"


@dataclass
class FieldSchema:
    """Schema definition for a single field."""
    name: str
    data_type: DataType
    confidence: float
    nullable: bool = True
    nested_schema: Optional[Dict[str, 'FieldSchema']] = None
    statistics: Dict[str, Any] = field(default_factory=dict)
    samples: List[Any] = field(default_factory=list)
    patterns: List[str] = field(default_factory=list)


@dataclass
class SchemaVersion:
    """Version control for schema evolution."""
    version: str
    schema: Dict[str, FieldSchema]
    timestamp: datetime
    hash: str
    parent_version: Optional[str] = None
    changes: List[str] = field(default_factory=list)


class TypeDetector:
    """AI-powered data type detection with confidence scoring."""
    
    def __init__(self):
        """Initialize type detector with pattern matchers."""
        self.patterns = {
            DataType.EMAIL: re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'),
            DataType.URL: re.compile(r'^https?://(?:www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b(?:[-a-zA-Z0-9()@:%_\+.~#?&/=]*)$'),
            DataType.DATE: re.compile(r'^\d{4}-\d{2}-\d{2}$'),
            DataType.DATETIME: re.compile(r'^\d{4}-\d{2}-\d{2}[T\s]\d{2}:\d{2}:\d{2}'),
        }
        
    def detect_type(self, values: List[Any], field_name: str = "") -> Tuple[DataType, float]:
        """
        Detect data type with confidence score.
        
        Args:
            values: List of values to analyze
            field_name: Optional field name for context
            
        Returns:
            Tuple of (detected_type, confidence_score)
        """
        if not values:
            return DataType.NULL, 1.0
            
        # Remove None values for analysis
        non_null_values = [v for v in values if v is not None]
        if not non_null_values:
            return DataType.NULL, 1.0
            
        null_ratio = 1 - (len(non_null_values) / len(values))
        
        # Type detection with confidence scoring
        type_scores = defaultdict(float)
        
        for value in non_null_values[:100]:  # Sample for performance
            str_value = str(value)
            
            # Check specific patterns first
            for dtype, pattern in self.patterns.items():
                if pattern.match(str_value):
                    type_scores[dtype] += 1.0
                    continue
                    
            # Check JSON/Object
            if isinstance(value, dict):
                type_scores[DataType.OBJECT] += 1.0
            elif isinstance(value, list):
                type_scores[DataType.ARRAY] += 1.0
            # Check boolean
            elif isinstance(value, bool) or str_value.lower() in ['true', 'false', 'yes', 'no']:
                type_scores[DataType.BOOLEAN] += 1.0
            # Check numeric
            elif isinstance(value, int) or str_value.isdigit():
                type_scores[DataType.INTEGER] += 1.0
            elif isinstance(value, float) or self._is_float(str_value):
                type_scores[DataType.FLOAT] += 1.0
            # Default to string
            else:
                type_scores[DataType.STRING] += 1.0
                
        if not type_scores:
            return DataType.UNKNOWN, 0.0
            
        # Get type with highest score
        detected_type = max(type_scores.items(), key=lambda x: x[1])
        confidence = detected_type[1] / len(non_null_values[:100])
        
        # Adjust confidence based on null ratio
        confidence *= (1 - null_ratio * 0.5)
        
        return detected_type[0], confidence
    
    def _is_float(self, value: str) -> bool:
        """Check if string represents a float."""
        try:
            float(value)
            return '.' in value
        except (ValueError, TypeError):
            return False


class SchemaInferenceEngine:
    """AI-driven schema inference engine that automatically detects
    and adapts to evolving data structures."""
    
    def __init__(self, 
                 similarity_threshold: float = 0.85,
                 min_confidence: float = 0.7):
        """Initialize the schema inference engine."""
        
        Args:
            similarity_threshold: Threshold for schema similarity matching
            min_confidence: Minimum confidence for type detection
        """
        self.similarity_threshold = similarity_threshold
        self.min_confidence = min_confidence
        self.type_detector = TypeDetector()
        self.schema_versions: Dict[str, SchemaVersion] = {} 
        self.current_version: Optional[str] = None
        
        # Load semantic model for field name similarity
        logger.info("Loading semantic similarity model...")
        self.semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        logger.info("Schema Inference Engine initialized")
    
    def infer_schema(self, 
                     data: List[Dict[str, Any]], 
                     source_id: str = "default") -> SchemaVersion:
        """Infer schema from a batch of unstructured data.
        
        Args:
            data: List of data records (dictionaries)
            source_id: Identifier for the data source
            
        Returns:
            SchemaVersion object with inferred schema
        """
        logger.info(f"Inferring schema for {len(data)} records from source: {source_id}")
        
        if not data:
            raise ValueError("Cannot infer schema from empty data")
        
        # Aggregate all fields across records
        field_values = defaultdict(list)
        for record in data:
            self._extract_fields(record, field_values)
        
        # Infer schema for each field
        schema = {}
        for field_name, values in field_values.items():
            schema[field_name] = self._infer_field_schema(field_name, values)
        
        # Create version
        schema_version = self._create_version(schema, source_id)
        
        # Check for schema drift
        if self.current_version:
            drift = self._detect_drift(schema_version)
            if drift:
                logger.warning(f"Schema drift detected: {drift}")
                schema_version.changes = drift
        
        # Store version
        self.schema_versions[schema_version.version] = schema_version
        self.current_version = schema_version.version
        
        logger.info(f"Schema inferred: version={schema_version.version}, "
                   f"fields={len(schema)}, confidence={self._average_confidence(schema):.2f}")
        
        return schema_version
    
    def _extract_fields(self, 
                       record: Dict[str, Any], 
                       field_values: Dict[str, List],
                       prefix: str = ""):
        """Recursively extract all fields from nested structures.
        
        Args:
            record: Data record
            field_values: Dictionary to accumulate field values
            prefix: Prefix for nested field names
        """
        for key, value in record.items():
            full_key = f"{prefix}.{key}" if prefix else key
            
            if isinstance(value, dict):
                # Nested object
                field_values[full_key].append(value)
                self._extract_fields(value, field_values, full_key)
            elif isinstance(value, list) and value and isinstance(value[0], dict):
                # Array of objects
                field_values[full_key].append(value)
                for item in value:
                    self._extract_fields(item, field_values, f"{full_key}[]")
            else:
                field_values[full_key].append(value)
    
    def _infer_field_schema(self, field_name: str, values: List[Any]) -> FieldSchema:
        """Infer schema for a single field.
        
        Args:
            field_name: Name of the field
            values: List of values for the field
            
        Returns:
            FieldSchema object
        """
        # Detect type
        data_type, confidence = self.type_detector.detect_type(values, field_name)
        
        # Calculate statistics
        non_null_values = [v for v in values if v is not None]
        statistics = {
            'total_count': len(values),
            'null_count': len(values) - len(non_null_values),
            'null_percentage': (len(values) - len(non_null_values)) / len(values) * 100,
            'unique_count': len(set(str(v) for v in non_null_values)),
        }
        
        # Add type-specific statistics
        if data_type in [DataType.INTEGER, DataType.FLOAT]:
            numeric_values = [float(v) for v in non_null_values if self._is_numeric(v)]
            if numeric_values:
                statistics.update({
                    'min': min(numeric_values),
                    'max': max(numeric_values),
                    'mean': np.mean(numeric_values),
                    'median': np.median(numeric_values),
                    'std': np.std(numeric_values),
                })
        elif data_type == DataType.STRING:
            str_values = [str(v) for v in non_null_values]
            if str_values:
                lengths = [len(s) for s in str_values]
                statistics.update({
                    'min_length': min(lengths),
                    'max_length': max(lengths),
                    'avg_length': np.mean(lengths),
                })
        
        # Extract patterns
        patterns = self._extract_patterns(non_null_values[:50])
        
        # Handle nested schemas
        nested_schema = None
        if data_type == DataType.OBJECT:
            # Recursively infer nested structure
            nested_data = [v for v in non_null_values if isinstance(v, dict)]
            if nested_data:
                nested_field_values = defaultdict(list)
                for obj in nested_data:
                    for k, v in obj.items():
                        nested_field_values[k].append(v)
                nested_schema = {
                    k: self._infer_field_schema(k, v) 
                    for k, v in nested_field_values.items()
                }
        
        return FieldSchema(
            name=field_name,
            data_type=data_type,
            confidence=confidence,
            nullable=statistics['null_count'] > 0,
            nested_schema=nested_schema,
            statistics=statistics,
            samples=non_null_values[:5],
            patterns=patterns
        )
    
    def _extract_patterns(self, values: List[Any]) -> List[str]:
        """Extract common patterns from values."""
        patterns = []
        str_values = [str(v) for v in values if v is not None]
        
        if not str_values:
            return patterns
        
        # Check for common patterns
        if all(v.isdigit() for v in str_values):
            patterns.append("all_digits")
        if all(v.isalpha() for v in str_values):
            patterns.append("all_alpha")
        if all(len(v) == len(str_values[0]) for v in str_values):
            patterns.append(f"fixed_length_{len(str_values[0])}")
        
        return patterns
    
    def _is_numeric(self, value: Any) -> bool:
        """Check if value is numeric."""
        try:
            float(value)
            return True
        except (ValueError, TypeError):
            return False
    
    def _create_version(self, schema: Dict[str, FieldSchema], source_id: str) -> SchemaVersion:
        """Create a new schema version."""
        # Generate version hash
        schema_dict = {k: self._field_to_dict(v) for k, v in schema.items()}
        schema_json = json.dumps(schema_dict, sort_keys=True)
        schema_hash = hashlib.sha256(schema_json.encode()).hexdigest()[:16]
        
        version_number = len(self.schema_versions) + 1
        version_id = f"v{version_number}_{schema_hash}"
        
        return SchemaVersion(
            version=version_id,
            schema=schema,
            timestamp=datetime.utcnow(),
            hash=schema_hash,
            parent_version=self.current_version
        )
    
    def _field_to_dict(self, field: FieldSchema) -> Dict:
        """Convert FieldSchema to dictionary."""
        return {
            'name': field.name,
            'type': field.data_type.value,
            'nullable': field.nullable,
            'confidence': field.confidence,
        }
    
    def _detect_drift(self, new_version: SchemaVersion) -> List[str]:
        """Detect schema drift between versions."""
        if not self.current_version:
            return []
        
        current = self.schema_versions[self.current_version]
        changes = []
        
        current_fields = set(current.schema.keys())
        new_fields = set(new_version.schema.keys())
        
        # Detect added fields
        added = new_fields - current_fields
        if added:
            changes.append(f"Added fields: {', '.join(added)}")
        
        # Detect removed fields
        removed = current_fields - new_fields
        if removed:
            changes.append(f"Removed fields: {', '.join(removed)}")
        
        # Detect type changes
        for field in current_fields & new_fields:
            current_type = current.schema[field].data_type
            new_type = new_version.schema[field].data_type
            if current_type != new_type:
                changes.append(f"Type changed for '{field}': {current_type.value} -> {new_type.value}")
        
        return changes
    
    def _average_confidence(self, schema: Dict[str, FieldSchema]) -> float:
        """Calculate average confidence across all fields."""
        if not schema:
            return 0.0
        return sum(f.confidence for f in schema.values()) / len(schema)
    
    def get_schema_similarity(self, version1: str, version2: str) -> float:
        """Calculate similarity between two schema versions using semantic analysis."""
        
        Args:
            version1: First schema version ID
            version2: Second schema version ID
            
        Returns:
            Similarity score (0-1)
        """
        if version1 not in self.schema_versions or version2 not in self.schema_versions:
            raise ValueError("Invalid schema version")
        
        schema1 = self.schema_versions[version1].schema
        schema2 = self.schema_versions[version2].schema
        
        # Get field names
        fields1 = set(schema1.keys())
        fields2 = set(schema2.keys())
        
        # Jaccard similarity for field names
        if not fields1 or not fields2:
            return 0.0
        
        intersection = len(fields1 & fields2)
        union = len(fields1 | fields2)
        jaccard_sim = intersection / union
        
        # Semantic similarity for field names
        if intersection > 0:
            common_fields = list(fields1 & fields2)
            embeddings1 = self.semantic_model.encode(common_fields)
            embeddings2 = self.semantic_model.encode(common_fields)
            
            # Cosine similarity
            from sklearn.metrics.pairwise import cosine_similarity
            semantic_sim = np.mean([
                cosine_similarity([embeddings1[i]], [embeddings2[i]])[0][0]
                for i in range(len(common_fields))
            ])
        else:
            semantic_sim = 0.0
        
        # Combined similarity
        return (jaccard_sim + semantic_sim) / 2
    
    def export_schema(self, version: Optional[str] = None) -> Dict:
        """Export schema in JSON format.
        
        Args:
            version: Schema version (uses current if not specified)
            
        Returns:
            Schema as dictionary
        """
        version = version or self.current_version
        if not version or version not in self.schema_versions:
            raise ValueError("No schema version available")
        
        schema_version = self.schema_versions[version]
        return {
            'version': schema_version.version,
            'timestamp': schema_version.timestamp.isoformat(),
            'hash': schema_version.hash,
            'fields': {
                name: self._field_to_export_dict(field)
                for name, field in schema_version.schema.items()
            }
        }
    
    def _field_to_export_dict(self, field: FieldSchema) -> Dict:
        """Convert field schema to export dictionary."""
        result = {
            'type': field.data_type.value,
            'nullable': field.nullable,
            'confidence': field.confidence,
            'statistics': field.statistics,
            'patterns': field.patterns,
        }
        
        if field.nested_schema:
            result['nested_fields'] = {
                name: self._field_to_export_dict(nested_field)
                for name, nested_field in field.nested_schema.items()
            }
        
        return result
