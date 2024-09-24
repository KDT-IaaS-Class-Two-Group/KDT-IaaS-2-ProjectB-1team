'use client';

import React from 'react';

export default function NavBar() {
  return (
    <nav className="w-full fixed top-0 left-0 z-50  ">
      <div className="container mx-auto flex justify-between items-center py-4 px-6">
        <div className="text-2xl font-bold">부제목</div>
      </div>
    </nav>
  );
}
