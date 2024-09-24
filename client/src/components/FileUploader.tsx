'use client';

import { useState } from 'react';
import { SpeedDial, SpeedDialAction, SpeedDialIcon } from '@mui/material';
import FileUploadIcon from '@mui/icons-material/FileUpload'; // 파일 업로드 아이콘

export default function FileUploader() {
  const [, setSelectedFile] = useState<File | null>(null);

  // 파일 선택 시 처리 함수
  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      setSelectedFile(file);
      uploadFile(file); // 파일 업로드 함수 호출
    }
  };

  // 파일 업로드 처리 함수
  const uploadFile = async (file: File) => {
    const formData = new FormData();
    formData.append('file', file);

    const response = await fetch('/api/upload', {
      method: 'POST',
      body: formData,
    });

    if (response.ok) {
      const result = await response.json();
      alert('File uploaded successfully');
      console.log(result.files);
    } else {
      alert('File upload failed');
    }
  };

  return (
    <div>
      {/* Material UI SpeedDial */}
      <SpeedDial
        ariaLabel="SpeedDial basic example"
        sx={{ position: 'absolute', bottom: 16, right: 16 }}
        icon={<SpeedDialIcon />}
      >
        <SpeedDialAction
          icon={<FileUploadIcon />}
          tooltipTitle="Upload File"
          onClick={() => {
            // 파일 선택 버튼 클릭 처리
            document.getElementById('file-input')?.click();
          }}
        />
      </SpeedDial>

      {/* 파일 선택 input */}
      <input
        type="file"
        id="file-input"
        style={{ display: 'none' }}
        onChange={handleFileChange}
      />
    </div>
  );
}
