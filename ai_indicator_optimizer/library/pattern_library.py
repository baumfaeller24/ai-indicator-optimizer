"""
Pattern Library - High-Performance CRUD Operations für Visual Trading Patterns
"""

import asyncio
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timezone, timedelta
from PIL import Image
import logging
from sqlalchemy import and_, or_, desc, func
from sqlalchemy.orm import Session
import uuid

from .models import VisualPattern, PerformanceMetrics
from .database import (
    DatabaseManager, get_database_manager, 
    VisualPatternModel, PatternPerformanceModel
)


class PatternSimilarityEngine:
    """
    Engine für Pattern-Ähnlichkeitssuche basierend auf Image Features
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def extract_image_features(self, image: Image.Image) -> np.ndarray:
        """
        Extrahiert Features aus Chart-Image für Ähnlichkeitsvergleich
        """
        # Konvertiere zu Grayscale für Feature-Extraktion
        gray_image = image.convert('L')
        
        # Resize für konsistente Feature-Größe
        resized = gray_image.resize((64, 64))
        
        # Einfache Histogram-Features (würde in Produktion durch CNN ersetzt)
        histogram = np.array(resized.histogram())
        
        # Normalisiere Features
        features = histogram / np.sum(histogram)
        
        return features
    
    def calculate_similarity(self, features1: np.ndarray, features2: np.ndarray) -> float:
        """
        Berechnet Ähnlichkeit zwischen zwei Feature-Vektoren
        """
        # Cosine Similarity
        dot_product = np.dot(features1, features2)
        norm1 = np.linalg.norm(features1)
        norm2 = np.linalg.norm(features2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        similarity = dot_product / (norm1 * norm2)
        return max(0.0, similarity)  # Clamp to [0, 1]


class PatternLibrary:
    """
    High-Performance Bibliothek für visuelle Trading-Patterns
    Optimiert für 191GB RAM und PostgreSQL
    """
    
    def __init__(self, db_manager: Optional[DatabaseManager] = None):
        self.db_manager = db_manager or get_database_manager()
        self.similarity_engine = PatternSimilarityEngine()
        self.logger = logging.getLogger(__name__)
        
        # Performance-Metriken
        self.query_count = 0
        self.cache_hits = 0
    
    def store_pattern(self, pattern: VisualPattern, performance: Optional[PerformanceMetrics] = None) -> str:
        """
        Speichert Pattern mit Performance-Metriken in Datenbank
        """
        try:
            with self.db_manager.get_session() as session:
                # Komprimiere Chart-Image
                image_data = self.db_manager.compress_image(pattern.chart_image)
                
                # Erstelle Database Model
                db_pattern = VisualPatternModel(
                    id=uuid.UUID(pattern.pattern_id) if pattern.pattern_id else uuid.uuid4(),
                    pattern_type=pattern.pattern_type,
                    symbol=pattern.market_context.get('symbol', 'UNKNOWN'),
                    timeframe=pattern.market_context.get('timeframe', '1m'),
                    chart_image_data=image_data,
                    chart_width=pattern.chart_image.width,
                    chart_height=pattern.chart_image.height,
                    confidence_score=pattern.confidence_score,
                    market_context=pattern.market_context,
                    performance_metrics=performance.__dict__ if performance else None,
                    win_rate=performance.win_rate if performance else None,
                    profit_factor=performance.profit_factor if performance else None,
                    max_drawdown=performance.max_drawdown if performance else None
                )
                
                session.add(db_pattern)
                session.commit()
                
                pattern_id = str(db_pattern.id)
                
                # Cache Pattern
                self.db_manager.cache_pattern(pattern_id, {
                    'pattern': pattern,
                    'performance': performance,
                    'db_model': db_pattern
                })
                
                self.logger.info(f"Stored pattern {pattern_id} of type {pattern.pattern_type}")
                return pattern_id
                
        except Exception as e:
            self.logger.error(f"Failed to store pattern: {e}")
            raise
    
    def get_pattern(self, pattern_id: str) -> Optional[VisualPattern]:
        """
        Lädt Pattern aus Datenbank oder Cache
        """
        # Prüfe Cache zuerst
        cached = self.db_manager.get_cached_pattern(pattern_id)
        if cached:
            self.cache_hits += 1
            return cached['pattern']
        
        try:
            with self.db_manager.get_session() as session:
                db_pattern = session.query(VisualPatternModel).filter(
                    VisualPatternModel.id == uuid.UUID(pattern_id)
                ).first()
                
                if not db_pattern:
                    return None
                
                # Dekomprimiere Image
                chart_image = self.db_manager.decompress_image(
                    db_pattern.chart_image_data,
                    db_pattern.chart_width,
                    db_pattern.chart_height
                )
                
                # Erstelle VisualPattern Object
                pattern = VisualPattern(
                    pattern_id=str(db_pattern.id),
                    pattern_type=db_pattern.pattern_type,
                    chart_image=chart_image,
                    confidence_score=db_pattern.confidence_score,
                    market_context=db_pattern.market_context,
                    performance_metrics=db_pattern.performance_metrics,
                    created_at=db_pattern.created_at
                )
                
                # Cache für zukünftige Zugriffe
                self.db_manager.cache_pattern(pattern_id, {'pattern': pattern})
                
                self.query_count += 1
                return pattern
                
        except Exception as e:
            self.logger.error(f"Failed to get pattern {pattern_id}: {e}")
            return None
    
    def query_similar_patterns(self, current_pattern: VisualPattern, 
                             similarity_threshold: float = 0.7,
                             limit: int = 10) -> List[Tuple[VisualPattern, float]]:
        """
        Findet ähnliche Patterns basierend auf Image-Features
        """
        try:
            # Extrahiere Features vom aktuellen Pattern
            current_features = self.similarity_engine.extract_image_features(current_pattern.chart_image)
            
            similar_patterns = []
            
            with self.db_manager.get_session() as session:
                # Query Patterns mit ähnlichem Typ und Context
                candidates = session.query(VisualPatternModel).filter(
                    and_(
                        VisualPatternModel.pattern_type == current_pattern.pattern_type,
                        VisualPatternModel.is_active == True,
                        VisualPatternModel.confidence_score >= 0.5
                    )
                ).order_by(desc(VisualPatternModel.confidence_score)).limit(limit * 3).all()
                
                for db_pattern in candidates:
                    try:
                        # Dekomprimiere Image für Feature-Extraktion
                        pattern_image = self.db_manager.decompress_image(
                            db_pattern.chart_image_data,
                            db_pattern.chart_width,
                            db_pattern.chart_height
                        )
                        
                        # Berechne Ähnlichkeit
                        pattern_features = self.similarity_engine.extract_image_features(pattern_image)
                        similarity = self.similarity_engine.calculate_similarity(
                            current_features, pattern_features
                        )
                        
                        if similarity >= similarity_threshold:
                            # Erstelle VisualPattern Object
                            pattern = VisualPattern(
                                pattern_id=str(db_pattern.id),
                                pattern_type=db_pattern.pattern_type,
                                chart_image=pattern_image,
                                confidence_score=db_pattern.confidence_score,
                                market_context=db_pattern.market_context,
                                performance_metrics=db_pattern.performance_metrics,
                                created_at=db_pattern.created_at
                            )
                            
                            similar_patterns.append((pattern, similarity))
                            
                    except Exception as e:
                        self.logger.warning(f"Error processing pattern {db_pattern.id}: {e}")
                        continue
            
            # Sortiere nach Ähnlichkeit
            similar_patterns.sort(key=lambda x: x[1], reverse=True)
            
            self.logger.info(f"Found {len(similar_patterns)} similar patterns")
            return similar_patterns[:limit]
            
        except Exception as e:
            self.logger.error(f"Similar pattern query failed: {e}")
            return []
    
    def update_pattern_performance(self, pattern_id: str, performance: PerformanceMetrics):
        """
        Aktualisiert Pattern-Performance in Datenbank
        """
        try:
            with self.db_manager.get_session() as session:
                db_pattern = session.query(VisualPatternModel).filter(
                    VisualPatternModel.id == uuid.UUID(pattern_id)
                ).first()
                
                if db_pattern:
                    db_pattern.performance_metrics = performance.__dict__
                    db_pattern.win_rate = performance.win_rate
                    db_pattern.profit_factor = performance.profit_factor
                    db_pattern.max_drawdown = performance.max_drawdown
                    db_pattern.updated_at = datetime.now(timezone.utc)
                    
                    session.commit()
                    
                    # Update Cache
                    cached = self.db_manager.get_cached_pattern(pattern_id)
                    if cached:
                        cached['performance'] = performance
                    
                    self.logger.info(f"Updated performance for pattern {pattern_id}")
                else:
                    self.logger.warning(f"Pattern {pattern_id} not found for performance update")
                    
        except Exception as e:
            self.logger.error(f"Failed to update pattern performance: {e}")
    
    def get_top_patterns(self, pattern_type: Optional[str] = None, 
                        symbol: Optional[str] = None,
                        timeframe: Optional[str] = None,
                        min_confidence: float = 0.7,
                        limit: int = 10) -> List[VisualPattern]:
        """
        Gibt Top-Performance Patterns zurück
        """
        try:
            with self.db_manager.get_session() as session:
                query = session.query(VisualPatternModel).filter(
                    VisualPatternModel.is_active == True,
                    VisualPatternModel.confidence_score >= min_confidence
                )
                
                # Filter anwenden
                if pattern_type:
                    query = query.filter(VisualPatternModel.pattern_type == pattern_type)
                
                if symbol:
                    query = query.filter(VisualPatternModel.symbol == symbol)
                
                if timeframe:
                    query = query.filter(VisualPatternModel.timeframe == timeframe)
                
                # Sortiere nach Performance
                db_patterns = query.order_by(
                    desc(VisualPatternModel.profit_factor),
                    desc(VisualPatternModel.win_rate),
                    desc(VisualPatternModel.confidence_score)
                ).limit(limit).all()
                
                patterns = []
                for db_pattern in db_patterns:
                    try:
                        chart_image = self.db_manager.decompress_image(
                            db_pattern.chart_image_data,
                            db_pattern.chart_width,
                            db_pattern.chart_height
                        )
                        
                        pattern = VisualPattern(
                            pattern_id=str(db_pattern.id),
                            pattern_type=db_pattern.pattern_type,
                            chart_image=chart_image,
                            confidence_score=db_pattern.confidence_score,
                            market_context=db_pattern.market_context,
                            performance_metrics=db_pattern.performance_metrics,
                            created_at=db_pattern.created_at
                        )
                        
                        patterns.append(pattern)
                        
                    except Exception as e:
                        self.logger.warning(f"Error loading pattern {db_pattern.id}: {e}")
                        continue
                
                self.logger.info(f"Retrieved {len(patterns)} top patterns")
                return patterns
                
        except Exception as e:
            self.logger.error(f"Failed to get top patterns: {e}")
            return []
    
    def get_pattern_statistics(self) -> Dict[str, Any]:
        """
        Gibt Pattern-Library Statistiken zurück
        """
        try:
            with self.db_manager.get_session() as session:
                # Gesamt-Statistiken
                total_patterns = session.query(func.count(VisualPatternModel.id)).scalar()
                active_patterns = session.query(func.count(VisualPatternModel.id)).filter(
                    VisualPatternModel.is_active == True
                ).scalar()
                
                # Pattern-Typen
                pattern_types = session.query(
                    VisualPatternModel.pattern_type,
                    func.count(VisualPatternModel.id)
                ).group_by(VisualPatternModel.pattern_type).all()
                
                # Performance-Statistiken
                avg_confidence = session.query(
                    func.avg(VisualPatternModel.confidence_score)
                ).filter(VisualPatternModel.is_active == True).scalar()
                
                avg_win_rate = session.query(
                    func.avg(VisualPatternModel.win_rate)
                ).filter(
                    and_(
                        VisualPatternModel.is_active == True,
                        VisualPatternModel.win_rate.isnot(None)
                    )
                ).scalar()
                
                return {
                    'total_patterns': total_patterns or 0,
                    'active_patterns': active_patterns or 0,
                    'pattern_types': dict(pattern_types) if pattern_types else {},
                    'avg_confidence': float(avg_confidence) if avg_confidence else 0.0,
                    'avg_win_rate': float(avg_win_rate) if avg_win_rate else 0.0,
                    'query_count': self.query_count,
                    'cache_hits': self.cache_hits,
                    'cache_hit_rate': self.cache_hits / max(1, self.query_count)
                }
                
        except Exception as e:
            self.logger.error(f"Failed to get pattern statistics: {e}")
            return {}
    
    def delete_pattern(self, pattern_id: str) -> bool:
        """
        Löscht Pattern aus Datenbank (Soft Delete)
        """
        try:
            with self.db_manager.get_session() as session:
                db_pattern = session.query(VisualPatternModel).filter(
                    VisualPatternModel.id == uuid.UUID(pattern_id)
                ).first()
                
                if db_pattern:
                    db_pattern.is_active = False
                    db_pattern.updated_at = datetime.now(timezone.utc)
                    session.commit()
                    
                    # Entferne aus Cache
                    if pattern_id in self.db_manager.pattern_cache:
                        del self.db_manager.pattern_cache[pattern_id]
                    
                    self.logger.info(f"Deleted pattern {pattern_id}")
                    return True
                else:
                    self.logger.warning(f"Pattern {pattern_id} not found for deletion")
                    return False
                    
        except Exception as e:
            self.logger.error(f"Failed to delete pattern {pattern_id}: {e}")
            return False
    
    async def bulk_import_patterns(self, patterns: List[Tuple[VisualPattern, Optional[PerformanceMetrics]]]) -> int:
        """
        Bulk-Import von Patterns für bessere Performance
        """
        try:
            imported_count = 0
            
            # Verwende Async Connection für Bulk Operations
            async_conn = await self.db_manager.get_async_connection()
            if not async_conn:
                # Fallback zu synchronem Import
                for pattern, performance in patterns:
                    try:
                        self.store_pattern(pattern, performance)
                        imported_count += 1
                    except Exception as e:
                        self.logger.warning(f"Failed to import pattern: {e}")
                return imported_count
            
            try:
                # Prepare bulk insert data
                insert_data = []
                for pattern, performance in patterns:
                    image_data = self.db_manager.compress_image(pattern.chart_image)
                    
                    insert_data.append({
                        'id': uuid.uuid4(),
                        'pattern_type': pattern.pattern_type,
                        'symbol': pattern.market_context.get('symbol', 'UNKNOWN'),
                        'timeframe': pattern.market_context.get('timeframe', '1m'),
                        'chart_image_data': image_data,
                        'chart_width': pattern.chart_image.width,
                        'chart_height': pattern.chart_image.height,
                        'confidence_score': pattern.confidence_score,
                        'market_context': pattern.market_context,
                        'performance_metrics': performance.__dict__ if performance else None,
                        'win_rate': performance.win_rate if performance else None,
                        'profit_factor': performance.profit_factor if performance else None,
                        'max_drawdown': performance.max_drawdown if performance else None,
                        'created_at': datetime.now(timezone.utc),
                        'updated_at': datetime.now(timezone.utc),
                        'is_active': True
                    })
                
                # Bulk insert
                await async_conn.executemany("""
                    INSERT INTO visual_patterns (
                        id, pattern_type, symbol, timeframe, chart_image_data,
                        chart_width, chart_height, confidence_score, market_context,
                        performance_metrics, win_rate, profit_factor, max_drawdown,
                        created_at, updated_at, is_active
                    ) VALUES (
                        $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16
                    )
                """, [
                    (
                        data['id'], data['pattern_type'], data['symbol'], data['timeframe'],
                        data['chart_image_data'], data['chart_width'], data['chart_height'],
                        data['confidence_score'], data['market_context'], data['performance_metrics'],
                        data['win_rate'], data['profit_factor'], data['max_drawdown'],
                        data['created_at'], data['updated_at'], data['is_active']
                    ) for data in insert_data
                ])
                
                imported_count = len(insert_data)
                self.logger.info(f"Bulk imported {imported_count} patterns")
                
            finally:
                await self.db_manager.async_pool.release(async_conn)
            
            return imported_count
            
        except Exception as e:
            self.logger.error(f"Bulk import failed: {e}")
            return 0