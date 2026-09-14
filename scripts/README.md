# Utilitários do repositório

Este diretório contém os utilitários aplicáveis ao repositório como um todo.

O script `validate_repository.sh` verifica o limite de tamanho de arquivos do GitHub, identifica caches ou arquivos temporários incluídos por engano e procura padrões comuns de credenciais antes de um commit.

Execução:

```bash
./scripts/validate_repository.sh
```
