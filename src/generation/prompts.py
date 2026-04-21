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

def build_prompt(query: str, context_chunks: list[dict]) -> str:
    # Retrieved chunks + questions = final prompt
    context_parts = []

    for i, chunk in enumerate(context_chunks, 1):
        meta = chunk["metadata"]
        similarity = chunk["similarity"]

        # Formating source label
        if meta.get("type") == "presentation":
            source = f"Prezentacija: {meta.get('source', '?')}, Slajd {meta.get('slide', '?')}"
        else:
            source = f"Knjiga: {meta.get('source', '?')}, Strana {meta.get('page', '?')}"

        context_parts.append(
            f"[Izvor {i} | {source} | relevantnost: {similarity}]\n"
            f"{chunk['content']}"
        )

    context_str = "\n\n---\n\n".join(context_parts)

    return f"""<context>
{context_str}
</context>

<question>
{query}
</question>

Odgovori na pitanje koristeći samo informacije iz konteksta iznad."""