from datapizza.agents import Agent
from datapizza.clients.openai import OpenAIClient
from datapizza.tools import tool
import spacy
from transformers import pipeline
from neo4j import GraphDatabase
import json

# =============================================================================
# TOOL 1: Text Summarization
# =============================================================================
@tool
def summarize_text(text: str, max_length: int = 130) -> str:
    """
    Riassume un testo lungo utilizzando modelli transformer.
    Perfetto per documenti, articoli e report.
    """
    try:
        summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
        
        # Gestisci testi molto lunghi dividendoli in chunk
        max_chunk = 1024
        chunks = [text[i:i+max_chunk] for i in range(0, len(text), max_chunk)]
        
        summaries = []
        for chunk in chunks:
            if len(chunk.split()) > 50:
                summary = summarizer(
                    chunk,
                    max_length=max_length,
                    min_length=30,
                    do_sample=False
                )
                summaries.append(summary[0]['summary_text'])
        
        final_summary = " ".join(summaries)
        compression_ratio = len(final_summary) / len(text) * 100
        
        # Ritorna una stringa formattata, NON JSON
        return (
            f"📝 SUMMARY:\n"
            f"Original length: {len(text)} characters\n"
            f"Summary length: {len(final_summary)} characters\n"
            f"Compression: {compression_ratio:.1f}%\n\n"
            f"{final_summary}"
        )
        
    except Exception as e:
        return f"Error during summarization: {str(e)}"


# =============================================================================
# TOOL 2: Named Entity Recognition
# =============================================================================
@tool
def extract_entities(text: str, language: str = "en") -> str:
    """
    Estrae entità nominate da un documento: persone, organizzazioni,
    luoghi, date, prodotti ed eventi.
    """
    # TODO: provare con modelli di Entity recogntion di HF
    try:
        # Carica il modello spaCy appropriato
        if language == "it":
            nlp = spacy.load("it_core_news_lg")
        else:
            nlp = spacy.load("en_core_web_lg")
        
        doc = nlp(text)
        
        # Estrai entità categorizzate
        entities_dict = {
            "PERSON": [],
            "ORG": [],
            "GPE": [],
            "LOC": [],
            "DATE": [],
            "MONEY": [],
            "PRODUCT": [],
            "EVENT": []
        }
        
        for ent in doc.ents:
            entity_type = ent.label_
            entity_text = ent.text.strip()
            
            if entity_type in entities_dict:
                if entity_text not in entities_dict[entity_type]:
                    entities_dict[entity_type].append(entity_text)
        
        # Rimuovi categorie vuote
        entities_dict = {k: v for k, v in entities_dict.items() if v}
        
        # Formatta output come stringa
        result = "🏷️  EXTRACTED ENTITIES:\n\n"
        
        total = sum(len(v) for v in entities_dict.values())
        result += f"Total entities found: {total}\n\n"
        
        for entity_type, entity_list in entities_dict.items():
            result += f"{entity_type} ({len(entity_list)}):\n"
            for entity in entity_list:
                result += f"  • {entity}\n"
            result += "\n"
        
        return result
        
    except Exception as e:
        return f"Error during entity extraction: {str(e)}"


# =============================================================================
# TOOL 3: Knowledge Graph Builder per Neo4j
# =============================================================================
@tool
def build_knowledge_graph(
    text: str,
    neo4j_uri: str = "bolt://localhost:7687",
    neo4j_user: str = "neo4j",
    neo4j_password: str = "password"
) -> str:
    """
    Costruisce un grafo di conoscenza in Neo4j dalle entità estratte.
    Crea nodi per entità e relazioni basate sulla co-occorrenza.
    """
    try:
        # Estrai entità dal testo
        nlp = spacy.load("en_core_web_lg")
        doc = nlp(text)
        
        # Connetti a Neo4j
        driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
        
        with driver.session() as session:
            # Cancella grafo precedente
            session.run("MATCH (n:Entity) DETACH DELETE n")
            
            nodes_created = 0
            relationships_created = 0
            
            # Crea nodi per ogni entità
            entity_map = {}
            for ent in doc.ents:
                entity_id = f"{ent.label_}_{ent.text.replace(' ', '_')}"
                
                if entity_id not in entity_map:
                    query = """
                    MERGE (n:Entity {
                        id: $id,
                        text: $text,
                        type: $type
                    })
                    RETURN n
                    """
                    session.run(query, id=entity_id, text=ent.text, type=ent.label_)
                    entity_map[entity_id] = ent
                    nodes_created += 1
            
            # Crea relazioni basate su co-occorrenza
            for sent in doc.sents:
                sent_entities = [ent for ent in sent.ents]
                
                for i, ent1 in enumerate(sent_entities):
                    for ent2 in sent_entities[i+1:]:
                        id1 = f"{ent1.label_}_{ent1.text.replace(' ', '_')}"
                        id2 = f"{ent2.label_}_{ent2.text.replace(' ', '_')}"
                        
                        query = """
                        MATCH (a:Entity {id: $id1})
                        MATCH (b:Entity {id: $id2})
                        MERGE (a)-[r:CO_OCCURS {context: $context}]->(b)
                        ON CREATE SET r.weight = 1
                        ON MATCH SET r.weight = r.weight + 1
                        RETURN r
                        """
                        session.run(query, id1=id1, id2=id2, context=sent.text[:200])
                        relationships_created += 1
            
            # Statistiche del grafo
            stats_query = """
            MATCH (n:Entity)
            OPTIONAL MATCH (n)-[r]-()
            RETURN 
                count(DISTINCT n) as node_count,
                count(DISTINCT r) as relationship_count
            """
            stats = session.run(stats_query).single()
        
        driver.close()
        
        result = (
            f"🕸️  KNOWLEDGE GRAPH CREATED:\n\n"
            f"Nodes created: {nodes_created}\n"
            f"Relationships created: {relationships_created}\n"
            f"Total nodes in graph: {stats['node_count']}\n"
            f"Total relationships: {stats['relationship_count']}\n\n"
            f"Neo4j URI: {neo4j_uri}\n"
            f"Query to visualize: MATCH (n:Entity) RETURN n LIMIT 25"
        )
        
        return result
        
    except Exception as e:
        return f"Error building knowledge graph: {str(e)}"


# =============================================================================
# ESEMPIO DI UTILIZZO CON AGENT
# =============================================================================
if __name__ == "__main__":
    
    # Documento di esempio
    sample_text = """
    Apple CEO Tim Cook announced the new iPhone 15 yesterday in Cupertino.
    The event took place at Apple Park with over 1000 journalists present.
    Microsoft and Google are closely watching Apple's moves in the smartphone market.
    The European Commission has opened an antitrust investigation against Meta Platforms.
    Mark Zuckerberg stated that he will invest 10 billion dollars in AI in 2024.
    """
    
    # Inizializza il client OpenAI
    client = OpenAIClient(
        api_key="YOUR_API_KEY",
        model="gpt-4o-mini"
    )
    
    # Crea un agent con tutti e tre i tool
    nlp_agent = Agent(
        name="nlp_knowledge_agent",
        client=client,
        tools=[summarize_text, extract_entities, build_knowledge_graph],
        system_prompt="""
        You are an advanced NLP and Knowledge Graph specialist.
        When asked to analyze a document, you should:
        1. First summarize the text
        2. Then extract named entities
        3. Finally build a knowledge graph in Neo4j
        
        Use the appropriate tools in sequence to provide comprehensive analysis.
        """
    )
    
    response = nlp_agent.run(
        f"Analyze this document and provide a complete NLP analysis with knowledge graph:\n\n{sample_text}"
    )
    
    print(response.text)
