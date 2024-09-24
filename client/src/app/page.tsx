'use client';

import React, { useState, useRef } from 'react';
import NavBar from '../components/NavBar';
import ImageGallery from '../components/ImageGallery';
import FloatingButton from '../components/FileUploader';

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
      <NavBar />
      {/* 네비게이션 바와 겹치지 않게 상단에 패딩 추가 */}
      <div className="pt-20">
        <ImageGallery images={images} />
      </div>

      <FloatingButton
        isLoading={isLoading}
        handleCancelUpload={handleCancelUpload}
        handleIconClick={handleIconClick}
      />

      {/* 숨겨진 파일 입력 */}
      <input
        type="file"
        ref={fileInputRef}
        className="hidden"
        onChange={handleFileChange}
      />
    </>
  );
}
