import json
import re

ARQUIVOS_TREINO = [
    "dataset_tcc_dataset1_reprovacoes_treino_346c1.jsonl",
    "dataset_tcc_dataset2_reprovacoes_treino_460c2.jsonl"
]

ARQUIVO_SAIDA = "exemplos_fewshot_selecionados.jsonl"

def carregar_dados(arquivos):
    dados = []
    for arquivo in arquivos:
        try:
            with open(arquivo, 'r', encoding='utf-8') as f:
                for linha in f:
                    dados.append(json.loads(linha))
        except FileNotFoundError:
            pass
    return dados

def calcular_jaccard(texto1, texto2):
    set1 = set(re.sub(r'[^\w\s]', '', str(texto1).lower()).split())
    set2 = set(re.sub(r'[^\w\s]', '', str(texto2).lower()).split())
    if not set1 or not set2: return 1.0
    return len(set1 & set2) / len(set1 | set2)

def possui_formatacao_proibida(texto):
    padrao_lista = r'\n\s*(?:[a-zA-Z]\)|\d+[\)\.]|-|•)\s'
    return bool(re.search(padrao_lista, texto))

def texto_tem_palavras_exatas(texto, palavras_chave):
    for palavra in palavras_chave:
        padrao = r'\b' + re.escape(palavra.lower()) + r'\b'
        if re.search(padrao, texto.lower()):
            return True
    return False

def selecionar_melhores_exemplos():
    dados = carregar_dados(ARQUIVOS_TREINO)
    
    candidatos = {
        "Estrutura Básica": [],
        "Propaganda": [],
        "Preconceito": [],
        "Erro Pedagógico": [],
        "Acessibilidade": []
    }

    for item in dados:
        _input = item.get("input", "")
        _output = item.get("output", "")
        
        tamanho_total = len(_input.split()) + len(_output.split())
        if tamanho_total > 600:
            continue
            
        # rejeitar apenas cópias (Jaccard > 0.80)
        if calcular_jaccard(_input, _output) > 0.80:
            continue
            
        if possui_formatacao_proibida(_output):
            continue
            
        if texto_tem_palavras_exatas(_input, ["formatação", "mancha", "pontuais", "revisão", "ortográfico"]):
            candidatos["Estrutura Básica"].append({"input": _input, "output": _output, "tamanho": tamanho_total})
                     
        palavras_preconceito = ["preconceito", "racismo", "estereótipo", "capacitismo", "discriminação"]
        if texto_tem_palavras_exatas(_input, palavras_preconceito) and texto_tem_palavras_exatas(_output, palavras_preconceito):
            candidatos["Preconceito"].append({"input": _input, "output": _output, "tamanho": tamanho_total})
            
        if texto_tem_palavras_exatas(_input, ["conceitual", "induz ao erro", "matemática", "didático", "pedagógico"]):
            candidatos["Erro Pedagógico"].append({"input": _input, "output": _output, "tamanho": tamanho_total})

        if texto_tem_palavras_exatas(_input, ["acessibilidade", "audiolivro", "html5", "audiodescrição", "libras", "leitor de tela"]):
            candidatos["Acessibilidade"].append({"input": _input, "output": _output, "tamanho": tamanho_total})

    textos_ja_usados = set()
    
    with open(ARQUIVO_SAIDA, 'w', encoding='utf-8') as f_out:
        for nome_criterio, lista_candidatos in candidatos.items():
            if lista_candidatos:
                lista_ordenada = sorted(lista_candidatos, key=lambda x: x["tamanho"])
                
                exemplo_escolhido = None
                for candidato in lista_ordenada:
                    if candidato["input"] not in textos_ja_usados:
                        exemplo_escolhido = candidato
                        break
                
                if exemplo_escolhido:
                    textos_ja_usados.add(exemplo_escolhido["input"])
                    
                    dicionario_saida = {
                        "criterio": nome_criterio,
                        "input": exemplo_escolhido["input"].strip(),
                        "output": exemplo_escolhido["output"].strip()
                    }
                    
                    f_out.write(json.dumps(dicionario_saida, ensure_ascii=False) + "\n")
                    print(f"✅ [{nome_criterio}] guardado! (Tamanho: {exemplo_escolhido['tamanho']} | Jaccard: {calcular_jaccard(exemplo_escolhido['input'], exemplo_escolhido['output']):.2f})")
                else:
                    print(f"⚠️ AVISO: Todos os exemplos para '{nome_criterio}' já foram usados.")
            else:
                print(f"⚠️ AVISO: Nenhum exemplo que cumpra TODAS as métricas foi encontrado para: {nome_criterio}")

if __name__ == "__main__":
    selecionar_melhores_exemplos()