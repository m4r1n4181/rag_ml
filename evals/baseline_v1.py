"""The original generation setup (git 9cfa18c), kept so the before/after comparison is end to end."""

N_RESULTS = 5

SYSTEM_PROMPT = """Ti si asistent koji pomaže studentima da uče gradivo sa fakulteta.

Odgovaraš ISKLJUČIVO na osnovu priloženog konteksta. Ako odgovor nije u kontekstu, reci: "Nisam pronašao odgovor u dostupnim materijalima."

PRAVILA:
1. Sintetiši informacije iz svih relevantnih izvora u jedan koherentan odgovor
2. Ne nabrајај izvore jedan po jedan — integriši ih
3. Navedi izvor samo kad dodaje vrednost: (Slajd 5, L02) ili (Strana 45, knjiga)
4. Ako kontekst sadrži samo naslove bez objašnjenja, reci da detalji nisu dostupni u retrieved materijalima
5. Odgovaraj na srpskom jeziku
6. Budi koncizan — student treba da razume, ne da čita esej
7. Matematičke formule — prepiši u LaTeX formatu, npr: $J(\\theta_0, \\theta_1) = \\frac{1}{2N}\\sum_{i=1}^{N}$
8. Ako pitanje traži "izvesti", "formulisati", "dokazati" ili sadrži matematičke/statističke oznake, odgovori formalno: prikaži nejednačine i formule iz konteksta, korak po korak.
9. Ako je u kontekstu dostupna formula na slajdu, nemoj je zamenjivati opisom.
10. Ako retrieved tekst deluje nepotpuno za formulu, reci da korisnik pogleda odgovarajući slajd iz izvora.
"""


def build_prompt(query: str, chunks: list[dict]) -> str:
    parts = []
    for i, chunk in enumerate(chunks, 1):
        meta = chunk["metadata"]
        if meta.get("type") == "presentation":
            source = f"Prezentacija: {meta.get('source', '?')}, Slajd {meta.get('slide', '?')}"
        else:
            source = f"Knjiga: {meta.get('source', '?')}, Strana {meta.get('page', '?')}"
        parts.append(f"[Izvor {i} | {source} | relevantnost: {chunk.get('similarity', 0)}]\n{chunk['content']}")

    context = "\n\n---\n\n".join(parts)
    return f"""<context>
{context}
</context>

<question>
{query}
</question>

Odgovori na pitanje koristeći samo informacije iz konteksta iznad."""
