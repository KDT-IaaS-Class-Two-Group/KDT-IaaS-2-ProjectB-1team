import { NextResponse } from 'next/server';
import { promises as fs } from 'fs';
import path from 'path';

// 루트 경로에 uploads 폴더 생성
const uploadDir = path.join(process.cwd(), 'uploads');

// 디렉토리가 없으면 생성
async function ensureDir(dir: string) {
  try {
    await fs.access(dir);
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  } catch (error) {
    await fs.mkdir(dir, { recursive: true });
  }
}

export async function POST(req: Request) {
  try {
    // uploads 폴더가 없으면 생성
    await ensureDir(uploadDir); 

    const formData = await req.formData();
    const file = formData.get('file') as File;

    if (!file) {
      return new NextResponse(JSON.stringify({ error: 'No file uploaded' }), { status: 400 });
    }

    // 파일 버퍼 생성 및 경로 지정
    const buffer = Buffer.from(await file.arrayBuffer());
    const filePath = path.join(uploadDir, file.name);

    // 파일 저장
    await fs.writeFile(filePath, buffer);

    return new NextResponse(JSON.stringify({ message: 'File uploaded successfully', filePath }), { status: 200 });
  } catch (error) {
    console.error('File upload error:', error);
    return new NextResponse(JSON.stringify({ error: 'File upload failed' }), { status: 500 });
  }
}
