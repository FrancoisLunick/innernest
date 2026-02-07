from sqlalchemy.orm import Session
from models.journal import JournalEntry

def create_journal_entry(db: Session, user_id: int, content: str, title: str | None = None, mood: str | None = None, tags: str | None = None):
    entry = JournalEntry(user_id = user_id, title = title, content = content, mood = mood, tags = tags)
    
    db.add(entry)
    db.commit()
    db.refresh(entry)
    
    return entry

def get_journal_entry(db: Session, entry_id: int) -> JournalEntry | None:
    return db.query(JournalEntry).filter(JournalEntry.id == entry_id).first()