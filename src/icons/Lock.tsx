import { Lock as LucideLock, LucideProps } from 'lucide-react';

const Lock = ({ className, ...props }: LucideProps) => {
  return <LucideLock className={className} {...props} />;
};

export default Lock;