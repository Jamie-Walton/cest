"use client"
import styles from '../app/home.module.css';
import {useRouter} from 'next/navigation';
import Image from 'next/image';

export default function Home() {
  const router = useRouter();
  return (
    <main>
      <nav className={`w-full pt-10 pb-5 px-16 flex items-center justify-between`}>
        <div className="text-xl font-semibold text-white" onClick={() => router.push('/')}>
        CEST Cardiac Analyzer
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
      <div className={styles.pageBackground}/>
    </main>
  );
}
