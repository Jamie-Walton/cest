"use client"
import React, { useState } from 'react';
import {useRouter} from 'next/navigation';
import Image from 'next/image';
import '../app/globals.css';

const NavBar = () => {
    const router = useRouter();
    return (
        <nav className={`w-full pt-10 pb-5 px-16 flex items-center justify-between bg-[#001EBC]`}
            style={{ boxShadow: '0px 1px 10px 0px rgba(0, 0, 0, 0.10)' }}>
            <div onClick={() => router.push('/')} className="cursor-pointer">
                <div className="text-xl font-semibold text-white" onClick={() => router.push('/')}>
                CEST Cardiac Analyzer
                </div>
            </div>
            <div className="flex space-x-4 h-[30px]">
                <button className="px-4 py-1 text-white transition-colors flex items-center justify-center">
                    <div className="flex space-x-2 justify-content items-center">
                        <p className="text-sm font-normal mr-2">researcher@berkeley.edu</p>
                        <Image
                            src="/icons/user-default.svg"
                            width={25}
                            height={25}
                            alt="User profile icon"
                        />
                    </div>
                </button>
            </div>
        </nav>
    );
};

export default NavBar;