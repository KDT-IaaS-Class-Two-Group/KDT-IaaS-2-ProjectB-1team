'use client';

import React, { useState, useRef } from 'react';
import CircularProgress from '@mui/material/CircularProgress'; // 로딩 스피너
import CloseIcon from '@mui/icons-material/Close'; // X 아이콘
import PhotoIcon from '@mui/icons-material/Photo'; // 기본 사진 아이콘

export default function Home() {
  const [isLoading, setIsLoading] = useState<boolean>(false); // 로딩 상태
  const [images, setImages] = useState<string[]>([]); // 업로드된 이미지들을 저장할 상태
  const fileInputRef = useRef<HTMLInputElement>(null);
  const controllerRef = useRef<AbortController | null>(null); // 업로드 취소용 AbortController
  const uploadTimeoutRef = useRef<NodeJS.Timeout | null>(null); // 업로드 시뮬레이션 취소용 Timeout Ref

  // 파일 선택 시 처리하는 함수
  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const files = event.target.files;
    if (files && files.length > 0) {
      setIsLoading(true); // 로딩 시작
      const fileArray = Array.from(files);

      // 업로드 요청 취소를 위한 AbortController 생성
      controllerRef.current = new AbortController();

      // 2초 후에 업로드가 완료되는 시뮬레이션
      uploadTimeoutRef.current = setTimeout(() => {
        const newImages = fileArray.map((file) => URL.createObjectURL(file)); // 파일 URL 생성
        setImages((prevImages) => [...prevImages, ...newImages]); // 기존 이미지에 새 이미지 추가
        setIsLoading(false); // 로딩 완료
      }, 2000);
    }
  };

  // 업로드 취소 함수
  const handleCancelUpload = () => {
    if (controllerRef.current) {
      controllerRef.current.abort(); // 업로드 요청 취소
    }

    if (uploadTimeoutRef.current) {
      clearTimeout(uploadTimeoutRef.current); // 시뮬레이션 중인 업로드 취소
    }

    setIsLoading(false); // 로딩 상태 종료
    console.log('업로드가 취소되었습니다.');
  };

  // 파일 선택 창 열기
  const handleIconClick = () => {
    if (fileInputRef.current) {
      fileInputRef.current.click();
    }
  };

  return (
    <>
      <div>
        {/* 투명한 상단 탭바 */}
        <nav className="w-full fixed top-0 left-0 z-50 ">
          <div className="container mx-auto flex justify-between items-center py-4 px-6">
            {/* 로고 */}
            <div className="text-2xl font-bold">부제목</div>

            {/* 네비게이션 메뉴 */}
            <ul className="flex space-x-8">
            </ul>
          </div>
        </nav>

        {/* 내용 영역 */}
        <div className="pt-16">
          {/* 기타 콘텐츠 */}
        </div>
      </div>

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

        {/* 업로드된 이미지 갤러리 */}
        <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-4 mt-8">
          {images.map((image, index) => (
            <div key={index} className="relative pb-[56.25%] h-0">
              <img
                src={image}
                alt={`uploaded-${index}`}
                className="absolute top-0 left-0 w-full h-full object-cover rounded-lg" />
            </div>
          ))}
        </div>
      </div>
    </>
  );
}
