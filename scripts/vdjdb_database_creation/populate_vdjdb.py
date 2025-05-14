import sqlite3
import json
import os

def create_tables(conn):
    cursor = conn.cursor()
    # Create complexes table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS complexes (
      complex_id       INTEGER PRIMARY KEY,
      mhc_a            TEXT     NOT NULL,
      mhc_b            TEXT     NOT NULL,
      mhc_class        TEXT     NOT NULL,
      antigen_epitope  TEXT     NOT NULL,
      antigen_gene     TEXT     NOT NULL,
      antigen_species  TEXT     NOT NULL,
      reference_id     TEXT     NOT NULL,
      method           TEXT     NOT NULL, -- JSON blob
      meta             TEXT     NOT NULL  -- JSON blob
    )
    """)

    # Create chains table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS chains (
      chain_id            INTEGER PRIMARY KEY AUTOINCREMENT,
      complex_id          INTEGER     NOT NULL REFERENCES complexes(complex_id),
      gene                TEXT        NOT NULL,
      cdr3                TEXT        NOT NULL,
      v_segm              TEXT        NOT NULL,
      j_segm              TEXT        NOT NULL,
      species             TEXT        NOT NULL,
      cdr3fix             TEXT        NOT NULL, -- JSON blob
      web_method          TEXT        NOT NULL,
      web_method_seq      TEXT        NOT NULL,
      web_cdr3fix_nc      TEXT        NOT NULL,
      web_cdr3fix_unmp    TEXT        NOT NULL,
      vdjdb_score         INTEGER     NOT NULL
    )
    """)
    conn.commit()

def main():
    input_txt_file = os.path.join('..', '..', 'data', 'vdjdb', 'vdjdb-2025-02-21', 'vdjdb.txt')  # Assumed to be in the same directory as the script or provide full path
    db_file = os.path.join('..', '..', 'databases', 'vdjdb', 'vdjdb.db') # Adjusted path relative to script location

    # Ensure the database directory exists
    os.makedirs(os.path.dirname(db_file), exist_ok=True)

    conn = None
    inserted_complexes = 0
    skipped_complexes = 0
    inserted_chains = 0
    error_rows = 0

    try:
        conn = sqlite3.connect(db_file)
        conn.execute("PRAGMA foreign_keys = ON")
        conn.enable_load_extension(True)
        try:
            # Attempt to load the JSON1 extension
            # The exact name of the shared library can vary by OS (e.g., json1.so, libjson1.dylib)
            # For built-in support (common in recent Python/SQLite versions), this might not be strictly needed
            # but including it for robustness if an older SQLite without built-in JSON1 is used.
            conn.load_extension('json1') 
        except sqlite3.OperationalError as e:
            print(f"Could not load JSON1 extension (might be built-in): {e}")
        conn.enable_load_extension(False)


        create_tables(conn)
        cursor = conn.cursor()

        # Keep track of inserted complex_ids to avoid duplicates
        existing_complex_ids = set()
        cursor.execute("SELECT complex_id FROM complexes")
        for row in cursor.fetchall():
            existing_complex_ids.add(row[0])

        with open(input_txt_file, 'r', encoding='utf-8') as f:
            header = f.readline().strip().split('\t') # Read and skip header
            expected_columns = [
                "complex.id", "gene", "cdr3", "v.segm", "j.segm", "species", 
                "mhc.a", "mhc.b", "mhc.class", "antigen.epitope", "antigen.gene", 
                "antigen.species", "reference.id", "method", "meta", "cdr3fix", 
                "web.method", "web.method.seq", "web.cdr3fix.nc", "web.cdr3fix.unmp", "vdjdb.score"
            ]
            
            # Verify header (optional, but good practice)
            if header != expected_columns:
                print(f"Warning: Input file header does not match expected columns.")
                print(f"Expected: {expected_columns}")
                print(f"Found:    {header}")
                # Decide if you want to stop or proceed if headers don't match
                # return 

            for line_number, line in enumerate(f, 2): # Start line_number from 2 due to header
                try:
                    values = line.strip().split('\t')
                    if len(values) != len(expected_columns):
                        print(f"Skipping row {line_number}: Expected {len(expected_columns)} columns, got {len(values)}. Line: '{line.strip()}'")
                        error_rows += 1
                        continue

                    row_data = dict(zip(expected_columns, values))

                    complex_id = int(row_data['complex.id'])

                    # Insert into complexes if not exists
                    if complex_id not in existing_complex_ids:
                        try:
                            cursor.execute("""
                            INSERT INTO complexes (complex_id, mhc_a, mhc_b, mhc_class, antigen_epitope, antigen_gene, antigen_species, reference_id, method, meta)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                            """, (
                                complex_id, row_data['mhc.a'], row_data['mhc.b'], row_data['mhc.class'],
                                row_data['antigen.epitope'], row_data['antigen.gene'], row_data['antigen.species'],
                                row_data['reference.id'], row_data['method'], row_data['meta']
                            ))
                            existing_complex_ids.add(complex_id)
                            inserted_complexes += 1
                        except sqlite3.IntegrityError: # Should not happen due to the check, but as a safeguard
                            skipped_complexes +=1
                        except Exception as e:
                            print(f"Error inserting complex_id {complex_id} (line {line_number}): {e}")
                            error_rows += 1
                            continue # Skip inserting chain if complex failed
                    else:
                        skipped_complexes += 1
                    
                    # Insert into chains
                    try:
                        cursor.execute("""
                        INSERT INTO chains (complex_id, gene, cdr3, v_segm, j_segm, species, cdr3fix, web_method, web_method_seq, web_cdr3fix_nc, web_cdr3fix_unmp, vdjdb_score)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, (
                            complex_id, row_data['gene'], row_data['cdr3'], row_data['v.segm'],
                            row_data['j.segm'], row_data['species'], row_data['cdr3fix'],
                            row_data['web.method'], row_data['web.method.seq'], row_data['web.cdr3fix.nc'],
                            row_data['web.cdr3fix.unmp'], int(row_data['vdjdb.score'])
                        ))
                        inserted_chains += 1
                    except Exception as e:
                        print(f"Error inserting chain for complex_id {complex_id} (line {line_number}): {e}")
                        error_rows += 1

                except ValueError as e:
                    print(f"Skipping row {line_number} due to data conversion error (e.g., non-integer for complex.id or vdjdb.score): {e}. Line: '{line.strip()}'")
                    error_rows += 1
                except Exception as e:
                    print(f"An unexpected error occurred processing line {line_number}: {e}. Line: '{line.strip()}'")
                    error_rows +=1
            
            conn.commit()

    except FileNotFoundError:
        print(f"Error: Input file '{input_txt_file}' not found. Please make sure it's in the same directory as the script or provide the correct path.")
        return
    except sqlite3.Error as e:
        print(f"SQLite error: {e}")
        if conn:
            conn.rollback()
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        if conn:
            conn.rollback()
    finally:
        if conn:
            conn.close()

    print("\n--- Summary ---")
    print(f"Complexes inserted: {inserted_complexes}")
    print(f"Complexes skipped (duplicates): {skipped_complexes}")
    print(f"Chains inserted: {inserted_chains}")
    print(f"Rows with errors/skipped: {error_rows}")
    print(f"Database created/updated at: {os.path.abspath(db_file)}")

if __name__ == '__main__':
    main() 