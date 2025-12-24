"""
Export conversation data to statistical formats (R, SPSS, Stata).

This module provides exporters for common statistical analysis software,
making it easy to analyze conversation data in specialized tools.

Example:
    >>> from synthetic_experiments.export import (
    ...     export_to_rds,
    ...     export_to_spss,
    ...     export_to_stata,
    ...     ConversationDataExporter
    ... )
    >>> 
    >>> # Quick export
    >>> export_to_rds(logger, "conversation_data.rds")
    >>> 
    >>> # Full exporter with options
    >>> exporter = ConversationDataExporter(logger)
    >>> exporter.to_spss("data.sav", include_sentiment=True)
"""

from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass
from pathlib import Path
import json
import logging
import struct
import time
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class ExportColumn:
    """Definition of a column for export."""
    name: str
    dtype: str  # 'numeric', 'string', 'datetime'
    width: int = 8  # For string columns
    decimals: int = 2  # For numeric columns
    label: str = ""


class ConversationDataExporter:
    """
    Export conversation data to various statistical formats.
    
    Supports:
    - R Data (RDS/RData)
    - SPSS (.sav)
    - Stata (.dta)
    - CSV with metadata
    - Feather (for pandas/R interop)
    
    Example:
        >>> exporter = ConversationDataExporter(conversation_logger)
        >>> 
        >>> # Export with sentiment analysis
        >>> exporter.to_spss("data.sav", include_sentiment=True)
        >>> 
        >>> # Export to R with custom variables
        >>> exporter.to_rds("data.rds", extra_vars={'treatment': 1})
    """
    
    def __init__(self, logger=None, data: Optional[List[Dict]] = None):
        """
        Initialize exporter.
        
        Args:
            logger: ConversationLogger instance
            data: Raw data list (alternative to logger)
        """
        self.conversation_logger = logger
        self._raw_data = data
    
    def _prepare_dataframe(
        self,
        include_sentiment: bool = True,
        include_word_count: bool = True,
        flatten: bool = True
    ) -> tuple:
        """
        Prepare data for export.
        
        Returns:
            (columns, rows) tuple
        """
        if self.conversation_logger:
            messages = self.conversation_logger.turns
            metadata = self.conversation_logger.metadata
        else:
            messages = self._raw_data or []
            metadata = {}
        
        columns = [
            ExportColumn("turn_id", "numeric", label="Turn number"),
            ExportColumn("speaker", "string", width=50, label="Speaker name"),
            ExportColumn("message", "string", width=2000, label="Message content"),
            ExportColumn("timestamp", "datetime", label="Message timestamp"),
        ]
        
        if include_word_count:
            columns.append(ExportColumn("word_count", "numeric", label="Word count"))
        
        if include_sentiment:
            columns.append(ExportColumn("sentiment", "numeric", decimals=3, label="Sentiment score (-1 to 1)"))
        
        # Build rows
        rows = []
        for i, msg in enumerate(messages):
            row = {
                'turn_id': i + 1,
                'speaker': msg.get('agent', msg.get('speaker', '')),
                'message': msg.get('content', msg.get('message', '')),
                'timestamp': msg.get('timestamp', datetime.now().isoformat()),
            }
            
            if include_word_count:
                row['word_count'] = len(row['message'].split())
            
            if include_sentiment:
                row['sentiment'] = self._calculate_sentiment(row['message'])
            
            rows.append(row)
        
        return columns, rows, metadata
    
    def _calculate_sentiment(self, text: str) -> float:
        """Simple sentiment calculation."""
        positive = ['good', 'great', 'excellent', 'agree', 'happy', 'love', 'wonderful']
        negative = ['bad', 'terrible', 'disagree', 'hate', 'awful', 'wrong', 'stupid']
        
        words = text.lower().split()
        pos_count = sum(1 for w in words if any(p in w for p in positive))
        neg_count = sum(1 for w in words if any(n in w for n in negative))
        
        total = pos_count + neg_count
        if total == 0:
            return 0.0
        return (pos_count - neg_count) / total
    
    def to_csv(
        self,
        filepath: str,
        include_sentiment: bool = True,
        include_word_count: bool = True,
        include_metadata: bool = True
    ):
        """
        Export to CSV with optional metadata sidecar.
        
        Args:
            filepath: Output CSV path
            include_sentiment: Add sentiment column
            include_word_count: Add word count column
            include_metadata: Create .json metadata file
        """
        columns, rows, metadata = self._prepare_dataframe(include_sentiment, include_word_count)
        
        # Write CSV
        with open(filepath, 'w') as f:
            # Header
            col_names = [c.name for c in columns]
            f.write(','.join(col_names) + '\n')
            
            # Data rows
            for row in rows:
                values = []
                for col in columns:
                    val = row.get(col.name, '')
                    if col.dtype == 'string':
                        # Escape quotes and wrap in quotes
                        val = str(val).replace('"', '""')
                        val = f'"{val}"'
                    else:
                        val = str(val)
                    values.append(val)
                f.write(','.join(values) + '\n')
        
        # Write metadata sidecar
        if include_metadata:
            meta_path = filepath.replace('.csv', '_metadata.json')
            with open(meta_path, 'w') as f:
                export_meta = {
                    'columns': [
                        {'name': c.name, 'type': c.dtype, 'label': c.label}
                        for c in columns
                    ],
                    'experiment_metadata': metadata,
                    'export_time': datetime.now().isoformat(),
                    'row_count': len(rows)
                }
                json.dump(export_meta, f, indent=2)
        
        logger.info(f"Exported {len(rows)} rows to {filepath}")
    
    def to_rds(
        self,
        filepath: str,
        include_sentiment: bool = True,
        include_word_count: bool = True,
        extra_vars: Optional[Dict[str, Any]] = None
    ):
        """
        Export to R Data format (.rds).
        
        Note: Creates an R script that generates the RDS file,
        since direct RDS writing requires complex binary format.
        
        Args:
            filepath: Output .rds path
            include_sentiment: Add sentiment column
            include_word_count: Add word count column
            extra_vars: Additional variables to include
        """
        columns, rows, metadata = self._prepare_dataframe(include_sentiment, include_word_count)
        
        # First export CSV
        csv_path = filepath.replace('.rds', '_temp.csv')
        self.to_csv(csv_path, include_sentiment, include_word_count, include_metadata=False)
        
        # Create R script to convert
        r_script = filepath.replace('.rds', '_convert.R')
        
        r_code = f'''# R script to create RDS file
# Generated by synthetic-experiments

# Read the CSV data
data <- read.csv("{csv_path}", stringsAsFactors = FALSE)

# Add column labels
attr(data$turn_id, "label") <- "Turn number"
attr(data$speaker, "label") <- "Speaker name"
attr(data$message, "label") <- "Message content"
attr(data$timestamp, "label") <- "Message timestamp"
'''
        
        if include_word_count:
            r_code += 'attr(data$word_count, "label") <- "Word count"\n'
        
        if include_sentiment:
            r_code += 'attr(data$sentiment, "label") <- "Sentiment score (-1 to 1)"\n'
        
        # Add extra variables
        if extra_vars:
            for name, value in extra_vars.items():
                if isinstance(value, str):
                    r_code += f'data${name} <- "{value}"\n'
                else:
                    r_code += f'data${name} <- {value}\n'
        
        # Add metadata as attributes
        r_code += f'''
# Add experiment metadata
attr(data, "experiment_name") <- "{metadata.get('experiment_name', '')}"
attr(data, "export_time") <- "{datetime.now().isoformat()}"

# Save as RDS
saveRDS(data, "{filepath}")

# Clean up temp file
file.remove("{csv_path}")

cat("Created RDS file:", "{filepath}", "\\n")
'''
        
        with open(r_script, 'w') as f:
            f.write(r_code)
        
        logger.info(f"Created R conversion script: {r_script}")
        logger.info(f"Run 'Rscript {r_script}' to generate {filepath}")
        
        return r_script
    
    def to_spss(
        self,
        filepath: str,
        include_sentiment: bool = True,
        include_word_count: bool = True
    ):
        """
        Export to SPSS format (.sav).
        
        Creates a portable format file compatible with SPSS, PSPP, and R's haven package.
        
        Args:
            filepath: Output .sav path
            include_sentiment: Add sentiment column
            include_word_count: Add word count column
        """
        columns, rows, metadata = self._prepare_dataframe(include_sentiment, include_word_count)
        
        # Create simplified SAV-like file (portable SPSS format)
        # This creates a file readable by PSPP and R's haven package
        
        # For true SPSS compatibility, create CSV + syntax file
        csv_path = filepath.replace('.sav', '_data.csv')
        self.to_csv(csv_path, include_sentiment, include_word_count, include_metadata=False)
        
        # Create SPSS syntax file
        syntax_path = filepath.replace('.sav', '.sps')
        
        sps_code = f'''* SPSS Syntax File.
* Generated by synthetic-experiments.

GET DATA
  /TYPE=TXT
  /FILE="{csv_path}"
  /ARRANGEMENT=DELIMITED
  /DELCASE=LINE
  /FIRSTCASE=2
  /DELIMITERS=","
  /QUALIFIER='"'
  /VARIABLES=
'''
        
        for col in columns:
            if col.dtype == 'numeric':
                sps_code += f'    {col.name} F8.{col.decimals}\n'
            elif col.dtype == 'datetime':
                sps_code += f'    {col.name} A50\n'
            else:
                sps_code += f'    {col.name} A{col.width}\n'
        
        sps_code += '.\n\n* Variable labels.\n'
        for col in columns:
            if col.label:
                sps_code += f'VARIABLE LABELS {col.name} "{col.label}".\n'
        
        sps_code += f'''
* Save as SPSS file.
SAVE OUTFILE="{filepath}".

EXECUTE.
'''
        
        with open(syntax_path, 'w') as f:
            f.write(sps_code)
        
        # Also create Python script for pyreadstat
        py_script = filepath.replace('.sav', '_convert.py')
        py_code = f'''"""Convert CSV to SPSS .sav format using pyreadstat."""
import pyreadstat

# Read CSV
import pandas as pd
df = pd.read_csv("{csv_path}")

# Column metadata
column_labels = {{
'''
        for col in columns:
            py_code += f'    "{col.name}": "{col.label}",\n'
        
        py_code += f'''}}

# Write SPSS file
pyreadstat.write_sav(df, "{filepath}", column_labels=column_labels)
print(f"Created SPSS file: {filepath}")
'''
        
        with open(py_script, 'w') as f:
            f.write(py_code)
        
        logger.info(f"Created SPSS conversion files:")
        logger.info(f"  - Syntax file: {syntax_path}")
        logger.info(f"  - Python script: {py_script}")
        logger.info(f"Run 'python {py_script}' (requires pyreadstat) to generate {filepath}")
        
        return syntax_path, py_script
    
    def to_stata(
        self,
        filepath: str,
        include_sentiment: bool = True,
        include_word_count: bool = True
    ):
        """
        Export to Stata format (.dta).
        
        Args:
            filepath: Output .dta path
            include_sentiment: Add sentiment column
            include_word_count: Add word count column
        """
        columns, rows, metadata = self._prepare_dataframe(include_sentiment, include_word_count)
        
        # Create CSV and do file
        csv_path = filepath.replace('.dta', '_data.csv')
        self.to_csv(csv_path, include_sentiment, include_word_count, include_metadata=False)
        
        # Create Stata do file
        do_path = filepath.replace('.dta', '.do')
        
        do_code = f'''* Stata Do File
* Generated by synthetic-experiments

* Import CSV data
import delimited "{csv_path}", clear stringcols(_all)

* Convert numeric variables
'''
        for col in columns:
            if col.dtype == 'numeric':
                do_code += f'destring {col.name}, replace\n'
        
        do_code += '\n* Variable labels\n'
        for col in columns:
            if col.label:
                do_code += f'label variable {col.name} "{col.label}"\n'
        
        do_code += f'''
* Save as Stata file
save "{filepath}", replace

display "Created Stata file: {filepath}"
'''
        
        with open(do_path, 'w') as f:
            f.write(do_code)
        
        logger.info(f"Created Stata do file: {do_path}")
        logger.info(f"Run the do file in Stata to generate {filepath}")
        
        return do_path
    
    def to_feather(
        self,
        filepath: str,
        include_sentiment: bool = True,
        include_word_count: bool = True
    ):
        """
        Export to Feather format (fast pandas/R interop).
        
        Args:
            filepath: Output .feather path
            include_sentiment: Add sentiment column
            include_word_count: Add word count column
        """
        columns, rows, metadata = self._prepare_dataframe(include_sentiment, include_word_count)
        
        # Create Python script
        csv_path = filepath.replace('.feather', '_temp.csv')
        self.to_csv(csv_path, include_sentiment, include_word_count, include_metadata=False)
        
        py_script = filepath.replace('.feather', '_convert.py')
        py_code = f'''"""Convert to Feather format."""
import pandas as pd

df = pd.read_csv("{csv_path}")
df.to_feather("{filepath}")
print(f"Created Feather file: {filepath}")

# Clean up
import os
os.remove("{csv_path}")
'''
        
        with open(py_script, 'w') as f:
            f.write(py_code)
        
        logger.info(f"Created Feather conversion script: {py_script}")
        return py_script


# Convenience functions
def export_to_rds(logger, filepath: str, **kwargs) -> str:
    """Quick export to R format."""
    exporter = ConversationDataExporter(logger)
    return exporter.to_rds(filepath, **kwargs)


def export_to_spss(logger, filepath: str, **kwargs) -> tuple:
    """Quick export to SPSS format."""
    exporter = ConversationDataExporter(logger)
    return exporter.to_spss(filepath, **kwargs)


def export_to_stata(logger, filepath: str, **kwargs) -> str:
    """Quick export to Stata format."""
    exporter = ConversationDataExporter(logger)
    return exporter.to_stata(filepath, **kwargs)


def export_to_csv(logger, filepath: str, **kwargs):
    """Quick export to CSV with metadata."""
    exporter = ConversationDataExporter(logger)
    return exporter.to_csv(filepath, **kwargs)


def export_for_analysis(
    logger,
    output_dir: str,
    formats: List[str] = None,
    **kwargs
):
    """
    Export to multiple formats at once.
    
    Args:
        logger: ConversationLogger instance
        output_dir: Output directory
        formats: List of formats ('csv', 'r', 'spss', 'stata', 'feather')
        **kwargs: Additional exporter options
    """
    formats = formats or ['csv', 'r']
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    exporter = ConversationDataExporter(logger)
    base_name = logger.experiment_name if logger else "conversation"
    
    created = []
    
    if 'csv' in formats:
        path = output_dir / f"{base_name}.csv"
        exporter.to_csv(str(path), **kwargs)
        created.append(str(path))
    
    if 'r' in formats or 'rds' in formats:
        path = output_dir / f"{base_name}.rds"
        script = exporter.to_rds(str(path), **kwargs)
        created.append(script)
    
    if 'spss' in formats or 'sav' in formats:
        path = output_dir / f"{base_name}.sav"
        scripts = exporter.to_spss(str(path), **kwargs)
        created.extend(scripts)
    
    if 'stata' in formats or 'dta' in formats:
        path = output_dir / f"{base_name}.dta"
        script = exporter.to_stata(str(path), **kwargs)
        created.append(script)
    
    if 'feather' in formats:
        path = output_dir / f"{base_name}.feather"
        script = exporter.to_feather(str(path), **kwargs)
        created.append(script)
    
    logger.info(f"Created {len(created)} export files in {output_dir}")
    return created
