from pathlib import PurePosixPath
from urllib.parse import quote

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Section(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    title: str = Field(min_length=1)
    text: str = Field(min_length=1)
    page_number: int | None = Field(default=None, gt=0)


class Chunk(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    doc_id: str = Field(min_length=1)
    source_path: str = Field(min_length=1)
    section_title: str = Field(min_length=1)
    page_number: int | None = Field(default=None, gt=0)
    section_index: int = Field(ge=0)
    chunk_index: int = Field(ge=0)
    chunk_id: str = Field(min_length=1)
    text: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_provenance(self) -> "Chunk":
        path = PurePosixPath(self.source_path)
        if (
            path.is_absolute()
            or ".." in path.parts
            or not path.parts
            or path.as_posix() != self.source_path
            or "\\" in self.source_path
            or ":" in self.source_path
        ):
            raise ValueError("source_path must be a normalized corpus-relative POSIX path")
        if self.doc_id != document_id(self.source_path):
            raise ValueError("doc_id does not match source_path")
        if self.chunk_id != f"{self.doc_id}#s{self.section_index}#c{self.chunk_index}":
            raise ValueError("chunk_id does not match its document, section, and chunk")
        return self

    @property
    def retrieval_text(self) -> str:
        return f"{self.section_title}\n{self.text}"


def document_id(source_path: str) -> str:
    return quote(source_path, safe="/-_.")
