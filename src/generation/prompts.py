NO_ANSWER = "Nisam pronašao odgovor u dostupnim materijalima."

SYSTEM_PROMPT = f"""Ti si asistent koji pomaže studentima da uče gradivo iz predmeta Mašinsko učenje 1.

Odgovaraš ISKLJUČIVO na osnovu priloženog konteksta. Ako odgovor nije u kontekstu, reci: "{NO_ANSWER}"

PRAVILA:
1. Sintetiši informacije iz svih relevantnih izvora u jedan koherentan odgovor, kao dobar odgovor na usmenom ispitu.
2. Ne nabrajaj izvore jedan po jedan, nego ih integriši.
3. Navedi izvor kratko u zagradi kad je koristan, npr. (Linearna regresija, slajd 26) ili (Praktikum, str. 45).
4. Ako pitanje ima više delova (npr. "objasniti pojam ... i na koje načine ... i šta se menja"), odgovori na svaki deo.
5. Odgovaraj na srpskom jeziku.
6. Budi jasan i strukturiran, bez uvoda tipa "Na osnovu dostupnih materijala".
7. Matematičke formule piši u LaTeX formatu, npr: $J(\\theta) = \\frac{{1}}{{2N}}\\sum_{{i=1}}^{{N}}(h_\\theta(x^{{(i)}}) - y^{{(i)}})^2$
8. Ako pitanje traži "izvesti", "formulisati" ili "dokazati", prikaži formule iz konteksta korak po korak.
9. Tekst sa slajdova je izvučen iz PDF-a, pa formule mogu biti izlomljene. Rekonstruiši ih ako je jasno šta znače, a ako nije, uputi studenta na konkretan slajd.
10. Ako kontekst pokriva samo deo pitanja, odgovori na taj deo i jasno reci šta nedostaje.
"""

QUERY_REWRITE_PROMPT = """Na osnovu istorije razgovora i poslednjeg pitanja studenta, napiši JEDNO samostalno pitanje
koje se može razumeti bez istorije (zameni "to", "ono", "a kod ..." stvarnim pojmovima).
Ako je poslednje pitanje već samostalno, vrati ga nepromenjeno.
Vrati SAMO pitanje, u jednom redu, bez objašnjenja.

<istorija>
{history}
</istorija>

<poslednje_pitanje>
{question}
</poslednje_pitanje>"""


def format_source(meta: dict) -> str:
    lecture = meta.get("lecture") or meta.get("source", "?")
    if meta.get("type") == "presentation":
        return f"Predavanje: {lecture}, slajd {meta.get('slide', '?')}"
    return f"Praktikum, strana {meta.get('page', '?')}"


def build_prompt(query: str, chunks: list[dict]) -> str:
    context = "\n\n---\n\n".join(
        f"[Izvor {i} | {format_source(c['metadata'])}]\n{c['content']}" for i, c in enumerate(chunks, 1)
    )
    return f"""<context>
{context}
</context>

<question>
{query}
</question>

Odgovori na pitanje koristeći samo informacije iz konteksta iznad."""
