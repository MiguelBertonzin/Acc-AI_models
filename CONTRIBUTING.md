# Manutenção do acervo

## Antes de adicionar uma nova rodada

1. Mantenha cada coleta ao lado do modelo e backend correspondentes.
2. Preserve o script exato, configuração, versão de ferramentas, seed e metadados de hardware.
3. Inclua dados brutos e um resumo legível (`README.md`, `RESULTADOS.md` ou `summary.json`).
4. Para overlays PYNQ, adicione `.bit` e `.hwh` com o mesmo nome-base e a documentação de endereços/interfaces.
5. Prefira manifests SHA-256 para pacotes de deploy e resultados finais.
6. Não versione caches, ambientes virtuais, datasets públicos integrais ou árvores temporárias das ferramentas.
7. Rode `./scripts/validate_repository.sh` antes do commit.

## Arquivos grandes

Arquivos de 50 a 100 MB geram aviso no GitHub e tornam clones mais pesados. Arquivos acima de 100 MB exigem Git LFS ou armazenamento externo. Antes de adotar LFS, confirme que as cotas da conta atendem ao volume e ao histórico esperado.

## Privacidade

Mesmo em repositório privado, não versione tokens, senhas, chaves SSH, cookies ou arquivos de credenciais. Caminhos absolutos presentes em relatórios são metadados históricos, não credenciais.

