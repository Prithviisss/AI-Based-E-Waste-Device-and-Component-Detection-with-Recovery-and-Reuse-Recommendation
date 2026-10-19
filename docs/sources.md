# Sources & Citations

All numeric values in the MCP knowledge base (`mcp_server/ewaste_mcp_server.py`)
must have a corresponding entry in this file before the system is considered complete.

> ✏ = placeholder — replace with real citation after verifying the figure.

---

## Primary Sources

### UN Global E-Waste Monitor
- **Author:** Forti, V., Baldé, C. P., Kuehr, R., Bel, G.
- **Title:** The Global E-waste Monitor 2024
- **Publisher:** United Nations University (UNU), ITU, ISWA
- **Year:** 2024
- **URL:** https://www.itu.int/en/ITU-D/Environment/Pages/Spotlight/Global-Ewaste-Monitor-2024.aspx
- **Access date:** ✏
- **Used for:** CO₂ saved per recovery route estimates; global e-waste volumes

### WEEE Directive
- **Title:** Directive 2012/19/EU of the European Parliament and of the Council on Waste Electrical and Electronic Equipment (WEEE)
- **Year:** 2012 (recast)
- **URL:** https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=celex%3A32012L0019
- **Used for:** Hazardous material classifications (Annex XIV), certified recycling requirements

### iFixit Repairability Scores / Component Guides
- **URL:** https://www.ifixit.com/
- **Used for:** Component resale value estimates, repairability guidance
- **Access date:** ✏
- **Note:** Verify specific guides per component (e.g. iPhone RAM, laptop SSD swap guides)

### WEEE Forum / Eurostat
- **URL:** https://www.weee-forum.org/
- **Used for:** Material recovery rates for metals (gold, copper, aluminium)

---

## Component-Level Citations

> Fill each row with the specific source used to derive the value in KNOWLEDGE_BASE.

| KB Key | Value field | Value | Source | URL / Section |
|--------|-------------|-------|--------|---------------|
| ram | resale_value_usd | 12.50 | ✏ iFixit / eBay completed listings (DDR4 8 GB avg) | ✏ |
| ram | co2_saved_direct_reuse_kg | 3.20 | ✏ UN E-waste Monitor 2024 | ✏ |
| ssd | resale_value_usd | 18.00 | ✏ | ✏ |
| ssd | co2_saved_direct_reuse_kg | 4.10 | ✏ | ✏ |
| pcb | material_value_usd | 3.50 | ✏ | ✏ |
| battery | hazard_note | "Lithium…" | WEEE Directive Annex XIV | https://eur-lex.europa.eu/... |
| cpu | resale_value_usd | 35.00 | ✏ | ✏ |
| gpu | resale_value_usd | 60.00 | ✏ | ✏ |
| mobile | co2_saved_direct_reuse_kg | 45.00 | ✏ UN E-waste Monitor 2024, Table ✏ | ✏ |
| computer | co2_saved_direct_reuse_kg | 300.00 | ✏ | ✏ |
| microwave | hazard_note | "Magnetron…" | ✏ | ✏ |

---

## RoHS Directive
- **Title:** Directive 2011/65/EU on the restriction of the use of certain hazardous substances in electrical and electronic equipment
- **URL:** https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:32011L0065
- **Used for:** Hazardous substance flags (lead, cadmium, mercury, hexavalent chromium, etc.)
