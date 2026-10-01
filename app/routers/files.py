import base64
from binascii import Error as Base64DecodeError
from urllib.parse import quote

from fastapi import APIRouter, Depends, Form, HTTPException, Request, Response, UploadFile, File
from sqlalchemy.orm import Session

from app.database import get_db
from app.schemas.schemas import FileOut
from app.repositories import repository
from app.services import file_service

router = APIRouter(tags=["Files"])


def _upload(
    session_id: str,
    filename: str,
    content: bytes,
    content_type: str | None,
    model_provider: str,
    model_name: str,
    db: Session,
    branch_id: str | None = None,
) -> object:
    mime_type = file_service.infer_mime_type(filename, content_type)
    content_data = file_service.encode_original(content)

    if file_service.is_image(filename):
        try:
            image_mime_type, image_data = file_service.encode_image(filename, content)
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e))
        return repository.save_file(
            db, session_id, filename, "", summary=None, branch_id=branch_id,
            file_type="image", mime_type=image_mime_type, content_data=content_data, image_data=image_data,
        )

    try:
        extracted = file_service.extract_text(filename, content)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    summary = file_service.generate_summary(filename, extracted, model_provider, model_name)
    return repository.save_file(
        db,
        session_id,
        filename,
        extracted,
        summary,
        branch_id,
        file_type="text",
        mime_type=mime_type,
        content_data=content_data,
    )


def _has_original_content(file) -> bool:
    return bool(file.content_data or (file.file_type == "image" and file.image_data))


def _with_content_url(request: Request, file):
    file.content_url = (
        str(request.url_for("get_file_content", file_id=file.id))
        if _has_original_content(file)
        else None
    )
    return file


def _with_content_urls(request: Request, files):
    return [_with_content_url(request, file) for file in files]


@router.post("/branches/{branch_id}/upload", response_model=FileOut, summary="브랜치에 파일 업로드")
async def upload_to_branch(
    branch_id: str,
    request: Request,
    file: UploadFile = File(...),
    model_provider: str = Form("openai"),
    model_name: str = Form("gpt-4o-mini"),
    db: Session = Depends(get_db),
):
    """브랜치 전용 파일을 업로드합니다. 이 브랜치의 채팅에서만 참조됩니다."""
    branch = repository.get_branch(db, branch_id)
    if branch is None:
        raise HTTPException(status_code=404, detail="branch를 찾을 수 없습니다.")
    content = await file.read()
    uploaded_file = _upload(
        branch.session_id,
        file.filename or "unknown",
        content,
        file.content_type,
        model_provider,
        model_name,
        db,
        branch_id,
    )
    return _with_content_url(request, uploaded_file)


@router.post("/sessions/{session_id}/upload", response_model=FileOut, summary="세션에 파일 업로드")
async def upload_to_session(
    session_id: str,
    request: Request,
    file: UploadFile = File(...),
    model_provider: str = Form("openai"),
    model_name: str = Form("gpt-4o-mini"),
    db: Session = Depends(get_db),
):
    """세션 전체 공유 파일을 업로드합니다. 이 세션의 모든 브랜치에서 참조됩니다."""
    content = await file.read()
    uploaded_file = _upload(
        session_id,
        file.filename or "unknown",
        content,
        file.content_type,
        model_provider,
        model_name,
        db,
        branch_id=None,
    )
    return _with_content_url(request, uploaded_file)


@router.get("/branches/{branch_id}/files", response_model=list[FileOut], summary="브랜치 전용 파일 목록")
def list_branch_files(branch_id: str, request: Request, db: Session = Depends(get_db)):
    branch = repository.get_branch(db, branch_id)
    if branch is None:
        raise HTTPException(status_code=404, detail="branch를 찾을 수 없습니다.")
    return _with_content_urls(request, repository.get_branch_files(db, branch_id))


@router.get("/sessions/{session_id}/files", response_model=list[FileOut], summary="세션 공유 파일 목록")
def list_session_files(session_id: str, request: Request, db: Session = Depends(get_db)):
    return _with_content_urls(request, repository.get_session_files(db, session_id))


@router.get("/files/{file_id}/content", name="get_file_content", summary="업로드 파일 원본 조회")
def get_file_content(file_id: str, db: Session = Depends(get_db)):
    """업로드된 원본 파일을 반환합니다. 이미지 미리보기와 원본 열기에 사용합니다."""
    uploaded_file = repository.get_file(db, file_id)

    if uploaded_file is None:
        raise HTTPException(status_code=404, detail="파일을 찾을 수 없습니다.")

    encoded_content = uploaded_file.content_data or (
        uploaded_file.image_data if uploaded_file.file_type == "image" else None
    )

    if not encoded_content:
        raise HTTPException(status_code=404, detail="파일 원본 데이터를 찾을 수 없습니다.")

    try:
        content = base64.b64decode(encoded_content)
    except Base64DecodeError:
        raise HTTPException(status_code=500, detail="파일 원본 데이터를 해석할 수 없습니다.")

    filename = quote(uploaded_file.filename)
    headers = {
        "Cache-Control": "private, max-age=604800",
        "Content-Disposition": f"inline; filename*=UTF-8''{filename}",
    }

    return Response(
        content=content,
        media_type=uploaded_file.mime_type or "application/octet-stream",
        headers=headers,
    )


@router.delete("/files/{file_id}", status_code=204, summary="파일 삭제")
def delete_file(file_id: str, db: Session = Depends(get_db)):
    """업로드된 파일을 삭제합니다."""
    ok = repository.delete_file(db, file_id)
    if not ok:
        raise HTTPException(status_code=404, detail="파일을 찾을 수 없습니다.")
