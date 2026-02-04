from datapizza.agents import Agent
from datapizza.clients.openai import OpenAIClient
from datapizza.tools import tool
from datapizza.modules.parsers.docling import DoclingParser
import sqlite3
import json
import os
from dotenv import load_dotenv

load_dotenv()

# 1. Setup database SQLite
def init_db():
    conn = sqlite3.connect('warehouse.db')
    cursor = conn.cursor()
    
    # Tabella magazzino
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS warehouse (
            product_id TEXT PRIMARY KEY,
            product_name TEXT,
            quantity INTEGER
        )
    ''')
    
    # Tabella elaborazioni
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS processing_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            pdf_filename TEXT,
            extracted_text TEXT,
            items_found TEXT,
            purchase_actions TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    conn.commit()
    conn.close()

# 2. Definisci i Tools per l'agent

@tool
def check_warehouse_stock(product_name: str) -> str:
    """
    Controlla la disponibilità di un prodotto nel magazzino.
    Args:
        product_name: Nome del prodotto da cercare
    Returns:
        JSON con disponibilità e quantità
    """
    conn = sqlite3.connect('warehouse.db')
    cursor = conn.cursor()
    
    cursor.execute(
        "SELECT product_name, quantity FROM warehouse WHERE product_name LIKE ?",
        (f"%{product_name}%",)
    )
    result = cursor.fetchone()
    conn.close()
    
    if result:
        return json.dumps({
            "available": True,
            "product": result[0],
            "quantity": result[1]
        })
    else:
        return json.dumps({
            "available": False,
            "product": product_name,
            "quantity": 0
        })

@tool
def simulate_purchase(product_name: str, quantity: int) -> str:
    """
    Simula l'acquisto di un prodotto e aggiorna il magazzino.
    Args:
        product_name: Nome del prodotto
        quantity: Quantità da acquistare
    Returns:
        Conferma dell'acquisto
    """
    conn = sqlite3.connect('warehouse.db')
    cursor = conn.cursor()
    
    # Controlla se esiste già
    cursor.execute("SELECT quantity FROM warehouse WHERE product_name = ?", (product_name,))
    existing = cursor.fetchone()
    
    if existing:
        new_quantity = existing[0] + quantity
        cursor.execute(
            "UPDATE warehouse SET quantity = ? WHERE product_name = ?",
            (new_quantity, product_name)
        )
    else:
        cursor.execute(
            "INSERT INTO warehouse (product_id, product_name, quantity) VALUES (?, ?, ?)",
            (f"PROD_{hash(product_name)}", product_name, quantity)
        )
    
    conn.commit()
    conn.close()
    
    return json.dumps({
        "success": True,
        "action": "purchased",
        "product": product_name,
        "quantity": quantity,
        "message": f"Acquistati {quantity} unità di {product_name}"
    })

@tool
def save_processing_result(pdf_filename: str, extracted_text: str, 
                          items_found: str, purchase_actions: str) -> str:
    """
    Salva il risultato dell'elaborazione nel database per batch processing.
    Args:
        pdf_filename: Nome del file PDF
        extracted_text: Testo estratto dal PDF
        items_found: Items trovati nel testo (JSON)
        purchase_actions: Azioni di acquisto effettuate (JSON)
    Returns:
        Conferma del salvataggio
    """
    conn = sqlite3.connect('warehouse.db')
    cursor = conn.cursor()
    
    cursor.execute(
        """INSERT INTO processing_results 
           (pdf_filename, extracted_text, items_found, purchase_actions)
           VALUES (?, ?, ?, ?)""",
        (pdf_filename, extracted_text, items_found, purchase_actions)
    )
    
    result_id = cursor.lastrowid
    conn.commit()
    conn.close()
    
    return json.dumps({
        "success": True,
        "result_id": result_id,
        "message": f"Risultato salvato con ID {result_id}"
    })

# 3. Setup Agent

def process_pdf(pdf_path: str):
    """
    Processa un PDF usando l'agent
    """
    # Parse PDF
    parser = DoclingParser()
    parsed_doc = parser.parse(pdf_path)
    extracted_text = parsed_doc.content
    
    # Setup OpenAI client e Agent
    client = OpenAIClient(
        api_key=os.getenv("OPENAI_API_KEY"),
        model="gpt-4o-mini"
    )
    
    agent = Agent(
        name="warehouse_manager",
        client=client,
        system_prompt="""
        Sei un assistente che gestisce ordini e magazzino.
        
        Il tuo compito:
        1. Analizza il testo fornito dall'utente (estratto da PDF)
        2. Identifica tutti i prodotti da comprare menzionati nel testo
        3. Per ogni prodotto:
           - Controlla se è disponibile in magazzino usando check_warehouse_stock
           - Se NON è disponibile o la quantità è insufficiente, 
             acquistalo usando simulate_purchase
        4. Alla fine, salva tutti i risultati usando save_processing_result
        
        Rispondi in modo strutturato con:
        - Lista prodotti trovati
        - Azioni di acquisto effettuate
        - Conferma del salvataggio
        """,
        tools=[
            check_warehouse_stock,
            simulate_purchase,
            save_processing_result
        ],
        max_steps=20  # Numero massimo di chiamate ai tools
    )
    
    # Esegui l'agent
    task = f"""
    Analizza questo testo estratto da un PDF e gestisci gli ordini:
    
    {extracted_text}
    
    File: {pdf_path}
    """
    
    response = agent.run(task)
    return response.text

# 4. Utilizzo
if __name__ == "__main__":
    init_db()
    
    # Esempio: popola warehouse con dati iniziali
    conn = sqlite3.connect('warehouse.db')
    cursor = conn.cursor()
    cursor.execute("INSERT OR IGNORE INTO warehouse VALUES ('P001', 'Penne', 100)")
    cursor.execute("INSERT OR IGNORE INTO warehouse VALUES ('P002', 'Matite', 50)")
    conn.commit()
    conn.close()
    
    # Processa un PDF
    result = process_pdf("data/doc1.pdf")
    print(result)
