# Treinamento FT UFAPE

Scripts para limpeza e preparacao de dados, treinamento e avaliacao de modelos de linguagem aplicados aos cenarios C1 e C2.

## Scripts principais

### Preparacao e selecao de dados

- `src/utils/clean_html_data.py`: limpa arquivos Excel, removendo HTML e corrigindo problemas de codificacao, e salva CSVs.
- `src/utils/clean_html_data_v2.py`: versao mais automatizada da limpeza de arquivos Excel, com descoberta dos arquivos de entrada e criacao da pasta de saida.
- `src/utils/merge_split_dataset.py`: mescla datasets e divide datasets em partes de treino e teste no formato JSONL.
- `src/few_shot.py`: seleciona exemplos representativos dos datasets para montar uma base de few-shot.

### Treinamento e exportacao

- `src/treino_mult_modelos_c1.py`: realiza fine-tuning com LoRA de varios modelos no dataset do cenario C1, usando validacao.
- `src/treino_mult_modelos_c2.py`: realiza fine-tuning com LoRA de varios modelos no dataset do cenario C2, usando validacao.
- `src/rodar_mult_treino_c1.sh`: submete o treinamento do cenario C1 ao SLURM, reservando GPU e recursos computacionais.
- `src/rodar_mult_treino_c2.sh`: submete o treinamento do cenario C2 ao SLURM, reservando GPU e recursos computacionais.
- `src/exportar_gguf.py`: converte um modelo ou checkpoint treinado para o formato GGUF quantizado.
- `src/rodar_exportacao.sh`: submete a conversao para GGUF ao SLURM.

### Geracao de metricas e graficos

- `src/semantica.py`: calcula similaridade semantica com SpaCy e SBERT nos CSVs de resultados.
- `src/generate_loss_training.py`: gera graficos agregados de training loss e validation loss por cenario, comparando os modelos.
- `src/generate_loss_training_per_model.py`: gera graficos individuais de training loss e validation loss para cada modelo e cenario.
- `src/gerar_boxplot.py`: cria boxplots de similaridade SBERT e cosseno para comparar os tres melhores modelos entre fine-tuning e prompt.

### Auditoria e analise estatistica

- `src/auditoria.py`: identifica alucinacoes, repeticoes e vazamentos de prompt, gerando resumos e separando casos validos e problematicos.
- `src/resultados_Finais.py`: calcula medias, medianas e desvios-padrao das metricas sobre os resultados higienizados.
- `src/teste_estatistico.py`: compara fine-tuning e prompt engineering com os testes de McNemar e Wilcoxon e gera um boxplot comparativo.

## Observacao

Os scripts usam caminhos relativos e, em alguns casos, caminhos absolutos configurados no proprio arquivo. Execute-os a partir do diretorio esperado pelo script e ajuste as variaveis de caminho quando necessario.
