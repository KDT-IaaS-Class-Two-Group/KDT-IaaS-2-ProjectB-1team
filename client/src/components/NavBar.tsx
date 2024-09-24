'use client';

import React from 'react';

export default function NavBar() {
  return (
    <nav className="w-full fixed top-0 left-0 z-50">
      <div className="container mx-auto flex justify-between items-center py-4 px-6">
        {/* 로고 */}
        <div className="text-2xl font-bold">부제목</div>

        {/* 네비게이션 메뉴 */}
        <ul className="flex space-x-8">
          {/* 네비게이션 메뉴 추가 */}
        </ul>
      </div>
    </nav>
  );
}
