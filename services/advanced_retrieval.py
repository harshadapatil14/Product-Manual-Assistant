"""
Advanced Retrieval Service for improving RAG performance
"""
import json
import os
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import re

class AdvancedRetrieval:
    """Service for advanced retrieval strategies to improve RAG performance"""
    
    def __init__(self):
        self.retrieval_strategies = {
            'hybrid': self._hybrid_retrieval,
            'rerank': self._rerank_retrieval,
            'multi_query': self._multi_query_expansion,
            'semantic_filter': self._semantic_filtering
        }
    
    def get_enhanced_context(self, query: str, chunks: List[str], strategy: str = 'hybrid') -> Dict:
        """
        Get enhanced context using advanced retrieval strategies
        
        Args:
            query: User's question
            chunks: Retrieved chunks
            strategy: Retrieval strategy to use
            
        Returns:
            Dictionary with enhanced context and metadata
        """
        if strategy not in self.retrieval_strategies:
            strategy = 'hybrid'
        
        # Apply the selected strategy
        enhanced_result = self.retrieval_strategies[strategy](query, chunks)
        
        return enhanced_result
    
    def _hybrid_retrieval(self, query: str, chunks: List[str]) -> Dict:
        """
        Hybrid retrieval combining semantic and keyword matching
        """
        # Extract keywords from query
        keywords = self._extract_keywords(query)
        
        # Score chunks based on multiple criteria
        scored_chunks = []
        for i, chunk in enumerate(chunks):
            score = self._calculate_hybrid_score(chunk, query, keywords)
            scored_chunks.append({
                'chunk': chunk,
                'score': score,
                'index': i,
                'keywords_found': self._count_keyword_matches(chunk, keywords)
            })
        
        # Sort by score and take top chunks
        scored_chunks.sort(key=lambda x: x['score'], reverse=True)
        top_chunks = scored_chunks[:min(5, len(scored_chunks))]
        
        # Combine top chunks with context
        enhanced_context = self._combine_chunks_with_context(top_chunks, query)
        
        return {
            'enhanced_context': enhanced_context,
            'strategy': 'hybrid',
            'chunks_analyzed': len(chunks),
            'top_chunks': len(top_chunks),
            'average_score': sum(c['score'] for c in top_chunks) / len(top_chunks) if top_chunks else 0,
            'keyword_coverage': self._calculate_keyword_coverage(top_chunks, keywords)
        }
    
    def _rerank_retrieval(self, query: str, chunks: List[str]) -> Dict:
        """
        Re-rank chunks based on relevance and quality
        """
        # Apply multiple ranking criteria
        reranked_chunks = []
        for i, chunk in enumerate(chunks):
            relevance_score = self._calculate_relevance_score(chunk, query)
            quality_score = self._calculate_quality_score(chunk)
            diversity_score = self._calculate_diversity_score(chunk, reranked_chunks)
            
            # Combined score with weights
            combined_score = (relevance_score * 0.5 + 
                            quality_score * 0.3 + 
                            diversity_score * 0.2)
            
            reranked_chunks.append({
                'chunk': chunk,
                'score': combined_score,
                'relevance': relevance_score,
                'quality': quality_score,
                'diversity': diversity_score,
                'index': i
            })
        
        # Sort by combined score
        reranked_chunks.sort(key=lambda x: x['score'], reverse=True)
        top_chunks = reranked_chunks[:min(4, len(reranked_chunks))]
        
        # Create enhanced context
        enhanced_context = self._create_structured_context(top_chunks, query)
        
        return {
            'enhanced_context': enhanced_context,
            'strategy': 'rerank',
            'chunks_analyzed': len(chunks),
            'top_chunks': len(top_chunks),
            'ranking_metrics': {
                'avg_relevance': sum(c['relevance'] for c in top_chunks) / len(top_chunks) if top_chunks else 0,
                'avg_quality': sum(c['quality'] for c in top_chunks) / len(top_chunks) if top_chunks else 0,
                'avg_diversity': sum(c['diversity'] for c in top_chunks) / len(top_chunks) if top_chunks else 0
            }
        }
    
    def _multi_query_expansion(self, query: str, chunks: List[str]) -> Dict:
        """
        Expand query with related terms and concepts
        """
        # Generate query variations
        query_variations = self._generate_query_variations(query)
        
        # Score chunks against each variation
        chunk_scores = {}
        for chunk in chunks:
            max_score = 0
            best_variation = query
            
            for variation in query_variations:
                score = self._calculate_similarity_score(chunk, variation)
                if score > max_score:
                    max_score = score
                    best_variation = variation
            
            chunk_scores[chunk] = {
                'score': max_score,
                'best_match': best_variation
            }
        
        # Select top chunks
        sorted_chunks = sorted(chunk_scores.items(), key=lambda x: x[1]['score'], reverse=True)
        top_chunks = sorted_chunks[:min(5, len(sorted_chunks))]
        
        # Create enhanced context with query variations
        enhanced_context = self._create_context_with_variations(top_chunks, query_variations)
        
        return {
            'enhanced_context': enhanced_context,
            'strategy': 'multi_query',
            'query_variations': query_variations,
            'chunks_analyzed': len(chunks),
            'top_chunks': len(top_chunks),
            'variation_coverage': len(set(c[1]['best_match'] for c in top_chunks))
        }
    
    def _semantic_filtering(self, query: str, chunks: List[str]) -> Dict:
        """
        Filter chunks based on semantic relevance
        """
        # Extract semantic concepts from query
        query_concepts = self._extract_semantic_concepts(query)
        
        # Filter and score chunks
        filtered_chunks = []
        for chunk in chunks:
            chunk_concepts = self._extract_semantic_concepts(chunk)
            semantic_overlap = self._calculate_concept_overlap(query_concepts, chunk_concepts)
            
            if semantic_overlap > 0.3:  # Threshold for relevance
                filtered_chunks.append({
                    'chunk': chunk,
                    'semantic_overlap': semantic_overlap,
                    'concepts': chunk_concepts
                })
        
        # Sort by semantic overlap
        filtered_chunks.sort(key=lambda x: x['semantic_overlap'], reverse=True)
        top_chunks = filtered_chunks[:min(4, len(filtered_chunks))]
        
        # Create context with semantic annotations
        enhanced_context = self._create_semantic_context(top_chunks, query_concepts)
        
        return {
            'enhanced_context': enhanced_context,
            'strategy': 'semantic_filter',
            'query_concepts': query_concepts,
            'chunks_analyzed': len(chunks),
            'filtered_chunks': len(filtered_chunks),
            'top_chunks': len(top_chunks),
            'avg_semantic_overlap': sum(c['semantic_overlap'] for c in top_chunks) / len(top_chunks) if top_chunks else 0
        }
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract important keywords from text"""
        # Simple keyword extraction (can be enhanced with NLP)
        words = re.findall(r'\b\w+\b', text.lower())
        # Filter out common stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those'}
        keywords = [word for word in words if word not in stop_words and len(word) > 2]
        return list(set(keywords))
    
    def _calculate_hybrid_score(self, chunk: str, query: str, keywords: List[str]) -> float:
        """Calculate hybrid score for a chunk"""
        # Keyword matching score
        keyword_matches = self._count_keyword_matches(chunk, keywords)
        keyword_score = keyword_matches / len(keywords) if keywords else 0
        
        # Semantic similarity score (simplified)
        semantic_score = self._calculate_similarity_score(chunk, query)
        
        # Length penalty (prefer medium-length chunks)
        length_penalty = 1.0 if 100 <= len(chunk) <= 500 else 0.8
        
        # Combined score
        return (keyword_score * 0.4 + semantic_score * 0.5 + length_penalty * 0.1)
    
    def _count_keyword_matches(self, text: str, keywords: List[str]) -> int:
        """Count keyword matches in text"""
        text_lower = text.lower()
        return sum(1 for keyword in keywords if keyword in text_lower)
    
    def _calculate_similarity_score(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts (simplified)"""
        # Simple word overlap similarity
        words1 = set(re.findall(r'\b\w+\b', text1.lower()))
        words2 = set(re.findall(r'\b\w+\b', text2.lower()))
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union)
    
    def _calculate_relevance_score(self, chunk: str, query: str) -> float:
        """Calculate relevance score for a chunk"""
        return self._calculate_similarity_score(chunk, query)
    
    def _calculate_quality_score(self, chunk: str) -> float:
        """Calculate quality score for a chunk"""
        # Factors: length, structure, completeness
        length_score = min(len(chunk) / 200, 1.0)  # Prefer chunks around 200 chars
        
        # Check for structured content (numbers, bullet points, etc.)
        structure_score = 0.5
        if re.search(r'\d+\.', chunk) or re.search(r'[-•*]', chunk):
            structure_score = 1.0
        
        # Check for complete sentences
        sentence_score = 0.5
        if chunk.strip().endswith(('.', '!', '?')):
            sentence_score = 1.0
        
        return (length_score * 0.4 + structure_score * 0.3 + sentence_score * 0.3)
    
    def _calculate_diversity_score(self, chunk: str, existing_chunks: List[Dict]) -> float:
        """Calculate diversity score to avoid redundancy"""
        if not existing_chunks:
            return 1.0
        
        # Calculate average similarity with existing chunks
        similarities = []
        for existing in existing_chunks:
            sim = self._calculate_similarity_score(chunk, existing['chunk'])
            similarities.append(sim)
        
        avg_similarity = sum(similarities) / len(similarities)
        return 1.0 - avg_similarity  # Higher diversity = lower similarity
    
    def _generate_query_variations(self, query: str) -> List[str]:
        """Generate query variations for better retrieval"""
        variations = [query]
        
        # Add variations based on common patterns
        query_lower = query.lower()
        
        # How-to variations
        if 'how' in query_lower:
            variations.extend([
                query.replace('how', 'what steps'),
                query.replace('how', 'procedure for'),
                query.replace('how', 'method to')
            ])
        
        # Problem variations
        if any(word in query_lower for word in ['error', 'problem', 'issue']):
            variations.extend([
                query.replace('error', 'solution'),
                query.replace('problem', 'fix'),
                query.replace('issue', 'resolve')
            ])
        
        # Definition variations
        if query_lower.startswith('what is'):
            variations.extend([
                query.replace('what is', 'define'),
                query.replace('what is', 'explain'),
                query.replace('what is', 'describe')
            ])
        
        return list(set(variations))  # Remove duplicates
    
    def _extract_semantic_concepts(self, text: str) -> List[str]:
        """Extract semantic concepts from text"""
        # Simplified concept extraction
        concepts = []
        
        # Extract technical terms (words with numbers, caps, etc.)
        tech_terms = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
        concepts.extend(tech_terms)
        
        # Extract measurements and specifications
        measurements = re.findall(r'\d+\s*(?:mm|cm|in|kg|lb|V|A|W|Hz)', text)
        concepts.extend(measurements)
        
        # Extract action words
        action_words = re.findall(r'\b(?:install|connect|configure|setup|test|check|verify|replace|repair)\b', text.lower())
        concepts.extend(action_words)
        
        return list(set(concepts))
    
    def _calculate_concept_overlap(self, concepts1: List[str], concepts2: List[str]) -> float:
        """Calculate overlap between concept lists"""
        if not concepts1 or not concepts2:
            return 0.0
        
        set1 = set(concepts1)
        set2 = set(concepts2)
        
        intersection = set1.intersection(set2)
        union = set1.union(set2)
        
        return len(intersection) / len(union)
    
    def _combine_chunks_with_context(self, chunks: List[Dict], query: str) -> str:
        """Combine chunks with additional context"""
        context_parts = []
        
        for i, chunk_data in enumerate(chunks, 1):
            chunk = chunk_data['chunk']
            score = chunk_data['score']
            keywords = chunk_data['keywords_found']
            
            context_parts.append(f"[Section {i} - Relevance: {score:.2f}, Keywords: {keywords}]\n{chunk}")
        
        return "\n\n".join(context_parts)
    
    def _create_structured_context(self, chunks: List[Dict], query: str) -> str:
        """Create structured context with metadata"""
        context_parts = []
        
        for i, chunk_data in enumerate(chunks, 1):
            chunk = chunk_data['chunk']
            relevance = chunk_data['relevance']
            quality = chunk_data['quality']
            
            context_parts.append(f"[Section {i} - Relevance: {relevance:.2f}, Quality: {quality:.2f}]\n{chunk}")
        
        return "\n\n".join(context_parts)
    
    def _create_context_with_variations(self, chunks: List[Tuple], variations: List[str]) -> str:
        """Create context with query variation information"""
        context_parts = []
        
        for i, (chunk, data) in enumerate(chunks, 1):
            best_match = data['best_match']
            score = data['score']
            
            context_parts.append(f"[Section {i} - Best Match: '{best_match}', Score: {score:.2f}]\n{chunk}")
        
        return "\n\n".join(context_parts)
    
    def _create_semantic_context(self, chunks: List[Dict], query_concepts: List[str]) -> str:
        """Create context with semantic annotations"""
        context_parts = []
        
        for i, chunk_data in enumerate(chunks, 1):
            chunk = chunk_data['chunk']
            overlap = chunk_data['semantic_overlap']
            concepts = chunk_data['concepts']
            
            context_parts.append(f"[Section {i} - Semantic Overlap: {overlap:.2f}, Concepts: {', '.join(concepts[:5])}]\n{chunk}")
        
        return "\n\n".join(context_parts)
    
    def _calculate_keyword_coverage(self, chunks: List[Dict], keywords: List[str]) -> float:
        """Calculate keyword coverage across chunks"""
        if not keywords:
            return 0.0
        
        covered_keywords = set()
        for chunk_data in chunks:
            chunk = chunk_data['chunk']
            chunk_lower = chunk.lower()
            for keyword in keywords:
                if keyword in chunk_lower:
                    covered_keywords.add(keyword)
        
        return len(covered_keywords) / len(keywords) 