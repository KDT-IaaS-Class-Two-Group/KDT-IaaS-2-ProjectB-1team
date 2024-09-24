'use client';

import React, { useRef } from 'react';
import CircularProgress from '@mui/material/CircularProgress'; // 로딩 스피너
import CloseIcon from '@mui/icons-material/Close'; // X 아이콘
import PhotoIcon from '@mui/icons-material/Photo'; // 기본 사진 아이콘

interface FileUploadButtonProps {
  isLoading: boolean;
  handleCancelUpload: () => void;
  handleFileChange: (event: React.ChangeEvent<HTMLInputElement>) => void;
}

export default function FileUploadButton({
  isLoading,
  handleCancelUpload,
  handleFileChange,
}: FileUploadButtonProps) {
  const fileInputRef = useRef<HTMLInputElement>(null);

  // 파일 선택 창 열기
  const handleIconClick = () => {
    if (fileInputRef.current) {
      fileInputRef.current.click();
    }
  };

  return (
    <div className="h-screen relative">
      {/* 오른쪽 아래에 원형 버튼으로 아이콘 표시 */}
      <div
        className="w-14 h-14 bg-blue-600 rounded-full flex items-center justify-center shadow-lg cursor-pointer absolute bottom-4 right-4"
        onClick={isLoading ? handleCancelUpload : handleIconClick} // 로딩 중일 때는 취소, 그렇지 않으면 파일 선택
      >
        {/* 업로드 상태에 따른 아이콘 변경 */}
        {isLoading ? (
          <div className="relative">
            <CircularProgress size={32} sx={{ color: '#fff' }} />
            <CloseIcon
              sx={{
                position: 'absolute',
                top: '50%',
                left: '50%',
                transform: 'translate(-50%, -50%)',
                fontSize: '24px',
                color: '#fff',
                cursor: 'pointer',
              }}
              onClick={handleCancelUpload} // X 버튼을 클릭하면 업로드 취소
            />
          </div>
        ) : (
          <PhotoIcon className="text-white text-3xl" />
        )}
      </div>

      {/* 숨겨진 파일 입력 */}
      <input
        type="file"
        ref={fileInputRef}
        className="hidden"
        onChange={handleFileChange} />
    </div>
  );
}
