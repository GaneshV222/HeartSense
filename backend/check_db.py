from app.database import engine
from sqlalchemy import text

def check():
    with engine.connect() as con:
        tables = con.execute(text("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'")).fetchall()
        table_names = [t[0] for t in tables]
        print("Existing tables:", table_names)
        for t in table_names:
            try:
                cnt = con.execute(text(f'SELECT COUNT(*) FROM "{t}"')).scalar()
                cols = con.execute(text(f"SELECT column_name, data_type FROM information_schema.columns WHERE table_name = '{t}'")).fetchall()
                print(f"Table {t} ({cnt} rows):")
                for c in cols:
                    print(f"  - {c[0]}: {c[1]}")
            except Exception as e:
                print(f"  Error on {t}: {e}")

if __name__ == "__main__":
    check()
