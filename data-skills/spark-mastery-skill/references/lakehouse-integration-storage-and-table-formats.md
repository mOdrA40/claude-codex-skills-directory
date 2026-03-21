# Lakehouse Integration, Storage, and Table Formats

## Rules

- Table format and storage design affect correctness, cost, and interoperability.
- Spark platform choices should align with transactionality, governance, and performance needs.
- Storage systems and table formats deserve architecture review, not default acceptance.
- Metadata scale and file-management behavior matter operationally.

## Practical Guidance

- Match table format to update/delete semantics and platform tooling.
- Monitor small files, metadata overhead, and compaction behavior.
- Design layout for common analytical and ETL patterns.
- Keep interoperability needs visible across teams and tools.

## Principal Review Lens

- Which storage or table-format choice creates the most hidden tax?
- Are we designing for real interoperability or future wish lists?
- What metadata or file-layout issue will hurt most at scale?
- Which platform standard most improves long-term reliability?
